"""Protocol configuration for umbrella-sampling runs."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

R_KCAL = 1.987204259e-3


@dataclass
class SystemPrep:
    force_field: str = "charmm36"
    water: str = "tip3p"
    ff_in_workdir: bool = True
    ignh: bool = True
    heavy_hydrogens: bool = False
    mass_repartition_factor: float = 3.0
    keep_hetatm: tuple[str, ...] = ()
    ph: float = 7.4
    cyclic_sb: float = 0.05
    cyclic_lb: float = 0.4
    cyclic_lb_conservative: float = 0.25
    cyclic_lb_from_manifest: bool = True
    box_edge_xy: float = 1.2
    box_edge_z: float = 1.0
    em_fmax_max: float = 5000.0
    pull_pbcatom: bool = True
    constraints: str = "all-bonds"   # dt 4 fs needs the heavy-atom bonds constrained too
    pull_box_margin: float = 0.04
    pull_headroom: float = 0.5
    ion_conc: float = 0.15
    pname: str = "NA"
    nname: str = "CL"


@dataclass
class SMD:
    rate: float = 0.001
    k: float = 1000.0
    time_ns: float = 2.0
    dt: float = 0.004
    restrain_target: bool = True


@dataclass
class Umbrella:
    window_spacing: float = 0.2
    dense_spacing: float = 0.1
    dense_until: float = 1.0
    k: float = 1000.0
    time_ns: float = 3.0
    extend_ns: float = 3.0
    max_ns: float = 24.0
    min_neff: float = 50.0
    max_drift_sd: float = 0.5
    equil_ns: float = 0.5
    discard_ns: float = 0.5
    max_distance: float = 2.0
    dt: float = 0.004
    barostat: str = "C-rescale"


@dataclass
class Wham:
    bootstraps: int = 200
    method: str = "b-hist"
    unit: str = "kCal"
    bins: int = 200
    tolerance: float = 1e-6
    temperature: float | None = None


@dataclass
class Run:
    backend: str = "chaperong"
    gmx: str = "gmx"
    chaperong: str = "run_CHAPERONg.sh"
    ntomp: int = 8
    ntmpi: int = 1
    cpu_cores: int = 0
    pin_threads: bool = True
    gpu: bool = False
    gpu_id: str = ""
    gpu_pme: bool = True
    gpu_bonded: bool = True
    gpu_update: bool = True
    nstlist: int = 100
    maxwarn: int = 2
    timeout_s: int = 172800
    window_workers: int = 1
    window_retries: int = 1
    retry_dt_scale: float = 0.5
    allow_failed_windows: int = 0
    skip_movie: bool = False
    movie_frames: int = 0
    pymol_headless: bool = True


@dataclass
class Analysis:
    plateau_width: float = 0.4
    overlap_min: float = 0.03
    jacobian_correction: bool = False
    exclude_roles: tuple[str, ...] = ("negative_control",)
    control_threshold: float = -5.0
    reference_temperature: float = 298.15
    estimator: str = "auto"          # wham | umbrella_integration | auto
    estimator_tolerance: float = 1.0  # kcal/mol RMS between the two, before UI is trusted


@dataclass
class Protocol:
    replicas: int = 3
    seed_base: int = 20181
    temperature: float = 310.0
    prep: SystemPrep = field(default_factory=SystemPrep)
    smd: SMD = field(default_factory=SMD)
    umbrella: Umbrella = field(default_factory=Umbrella)
    wham: Wham = field(default_factory=Wham)
    run: Run = field(default_factory=Run)
    analysis: Analysis = field(default_factory=Analysis)

    @classmethod
    def load(cls, path: str | Path | None) -> "Protocol":
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        return _from_dict(cls, data)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True))

    @property
    def kT_kcal(self) -> float:
        return R_KCAL * self.temperature

    def validate(self) -> list[str]:
        """Cheap consistency checks; raises on anything that would corrupt results."""
        problems = []
        big_dt = max(self.smd.dt, self.umbrella.dt)
        mrf = self.prep.mass_repartition_factor
        if big_dt > 0.0025 and not (mrf >= 2.5 or self.prep.heavy_hydrogens):
            problems.append(
                f"dt = {big_dt} ps needs hydrogen mass repartitioning: set "
                "prep.mass_repartition_factor to 3 (gmx scales H to 3x and refuses "
                "any bound atom that would drop below that mass)"
            )
        if self.prep.heavy_hydrogens and mrf > 1:
            problems.append(
                "prep.heavy_hydrogens (pdb2gmx -heavyh) and prep.mass_repartition_factor "
                "are two ways to do the same thing; -heavyh has no lower-mass guard and "
                "leaves methyl carbons at ~2.9 amu, lighter than their own hydrogens"
            )
        if big_dt > 0.0035 and self.prep.constraints != "all-bonds":
            problems.append(
                f"dt = {big_dt * 1000:.0f} fs leaves heavy-atom bonds unconstrained: a "
                "carboxylate C=O oscillates with a ~22 fs period, so 4 fs gives five steps "
                "per period and a window integrates into a nan every few hundred windows. "
                "Set prep.constraints to all-bonds, or drop dt to 0.003"
            )
        if self.umbrella.discard_ns >= self.umbrella.time_ns:
            problems.append("umbrella.discard_ns must be shorter than umbrella.time_ns")
        if self.umbrella.dense_spacing > self.umbrella.window_spacing:
            problems.append("umbrella.dense_spacing should be finer than window_spacing")
        if self.analysis.plateau_width >= self.umbrella.max_distance / 2:
            problems.append("analysis.plateau_width is more than half the reaction coordinate")
        if problems:
            raise ValueError("protocol is inconsistent:\n  - " + "\n  - ".join(problems))
        return problems


def _from_dict(cls, data: dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) or (isinstance(value, dict) and hasattr(f.default_factory, "__call__")
                                    and is_dataclass(f.default_factory())):
            kwargs[f.name] = _from_dict(type(f.default_factory()), value)
        elif isinstance(f.default, tuple):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
