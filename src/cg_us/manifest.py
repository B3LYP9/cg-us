"""Manifest parsing and experimental-affinity bookkeeping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

R_KCAL = 1.987204259e-3
DEFAULT_T = 298.15

REQUIRED = ("pdb", "name", "target", "binder")


@dataclass
class Entry:
    name: str
    pdb: Path
    target: str
    binder: str
    role: str = "benchmark"
    sim_ph: float = 7.4
    restrain_binder: bool = False
    dg_exp: float | None = None
    dg_exp_source: str = ""
    affinity_type: str = ""
    assay_T: float = DEFAULT_T
    cyclic_type: str = ""
    binder_seq: str = ""
    binder_length: int | None = None
    caveats: str = ""

    @property
    def in_regression(self) -> bool:
        return self.dg_exp is not None and self.role != "negative_control"

    @property
    def head_to_tail(self) -> bool | None:
        """True/False when the manifest states the cyclisation, None when silent."""
        c = self.cyclic_type.lower()
        if not c:
            return None
        if "head-to-tail" in c or "head to tail" in c or "backbone" in c:
            return True
        return False


def dg_from_kd(kd_molar: float, temperature: float = DEFAULT_T) -> float:
    return R_KCAL * temperature * math.log(kd_molar)


def read_manifest(path: str | Path, root: str | Path | None = None) -> list[Entry]:
    path = Path(path)
    root = Path(root) if root else path.parent
    df = pd.read_csv(path, dtype=str, keep_default_na=False).replace("", pd.NA)
    df = df.dropna(how="all")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"manifest {path} is missing required column(s): {', '.join(missing)}")

    entries: list[Entry] = []
    for i, row in df.iterrows():
        pdb = root / str(row["pdb"])
        if not pdb.exists():
            raise FileNotFoundError(f"row {i}: structure not found: {pdb}")

        assay_T = _float(row.get("assay_T_K")) or DEFAULT_T
        dg, source = _resolve_dg(row, assay_T)

        entries.append(
            Entry(
                name=str(row["name"]).strip(),
                pdb=pdb,
                target=str(row["target"]).strip(),
                binder=str(row["binder"]).strip(),
                role=_str(row.get("role")) or "benchmark",
                sim_ph=_float(row.get("sim_ph")) or 7.4,
                restrain_binder=_bool(row.get("restrain_binder")),
                dg_exp=dg,
                dg_exp_source=source,
                affinity_type=_str(row.get("affinity_type")),
                assay_T=assay_T,
                cyclic_type=_str(row.get("cyclic_type")),
                binder_seq=_str(row.get("binder_seq")),
                binder_length=int(_float(row.get("binder_length")) or 0) or None,
                caveats=_str(row.get("caveats")),
            )
        )

    names = [e.name for e in entries]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate system names in manifest: {', '.join(sorted(dupes))}")
    return entries


def _resolve_dg(row, assay_T: float) -> tuple[float | None, str]:
    dg = _float(row.get("dg_exp_kcal_298K"))
    if dg is not None:
        return dg, "manifest"
    kd = _float(row.get("kd_exp_M"))
    if kd is not None and kd > 0:
        return dg_from_kd(kd, assay_T), f"derived from Kd={kd:g} M at {assay_T:g} K"
    return None, ""


def _float(v) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def _str(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return str(v).strip()


def _bool(v) -> bool:
    s = _str(v).lower()
    return s in {"1", "true", "yes", "y"}
