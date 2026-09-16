"""WHAM execution and PMF post-processing."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .backends.base import Gmx
from .config import Protocol
from .xvg import read_xvg


@dataclass
class PMF:
    xi: np.ndarray
    energy: np.ndarray
    error: np.ndarray | None = None

    def shifted(self) -> "PMF":
        return PMF(self.xi, self.energy - self.energy.min(), self.error)


def run_wham(workdir: str | Path, proto: Protocol, prefix: str = "wham",
             begin_ps: float | None = None, end_ps: float | None = None,
             bootstraps: int | None = None) -> dict:
    workdir = Path(workdir)
    out = workdir / "analysis"
    out.mkdir(exist_ok=True)
    n_bs = proto.wham.bootstraps if bootstraps is None else bootstraps
    temp = proto.wham.temperature or proto.temperature

    args = [
        "wham",
        "-it", "tpr_files.dat",
        "-if", "pullf_files.dat",
        "-o", f"analysis/{prefix}_pmf.xvg",
        "-hist", f"analysis/{prefix}_hist.xvg",
        "-unit", proto.wham.unit,
        "-temp", str(temp),
        "-bins", str(proto.wham.bins),
        "-tol", str(proto.wham.tolerance),
    ]
    if begin_ps is not None:
        args += ["-b", str(begin_ps)]
    if end_ps is not None:
        args += ["-e", str(end_ps)]
    if n_bs:
        args += ["-nBootstrap", str(n_bs), "-bs-method", proto.wham.method,
                 "-bsres", f"analysis/{prefix}_bsres.xvg",
                 "-bsprof", f"analysis/{prefix}_bsprof.xvg"]

    Gmx(proto.run.gmx, workdir, out / f"{prefix}_wham.log").run(args)

    return {
        "pmf": out / f"{prefix}_pmf.xvg",
        "hist": out / f"{prefix}_hist.xvg",
        "bsres": out / f"{prefix}_bsres.xvg" if n_bs else None,
        "bsprof": out / f"{prefix}_bsprof.xvg" if n_bs else None,
    }


def load_pmf(paths: dict) -> PMF:
    data, _ = read_xvg(paths["pmf"])
    xi, energy = data[:, 0], data[:, 1]
    err = None
    bsres = paths.get("bsres")
    if bsres and Path(bsres).exists():
        bs, _ = read_xvg(bsres)
        if bs.shape[1] >= 3 and len(bs) == len(xi):
            err = bs[:, 2]
    finite = np.isfinite(energy)
    return PMF(xi[finite], energy[finite], err[finite] if err is not None else None)


def load_histograms(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data, _ = read_xvg(path)
    return data[:, 0], data[:, 1:]


def connected_range(overlap: dict, threshold: float) -> float | None:
    """Where the window ladder stops carrying information.

    WHAM fixes the offset between two windows from the samples they share. A
    neighbouring pair with no shared samples leaves that offset unconstrained,
    so everything beyond it is displaced by an arbitrary constant - which is
    what put 8 kcal/mol steps into the first campaign's profiles. The connected
    stretch ends at the centre of the last window still tied to its neighbour.
    """
    overlaps = overlap.get("overlaps") or []
    centers = overlap.get("window_centers_nm") or []
    if not overlaps or len(centers) != len(overlaps) + 1:
        return None
    for i, o in enumerate(overlaps):
        if o < threshold:
            return float(centers[i])
    return None


def binding_free_energy(pmf: PMF, plateau_width: float = 0.4,
                        jacobian: bool = False, kT: float | None = None,
                        xi_max: float | None = None) -> dict:
    xi, g, err_of_xi = pmf.xi, pmf.energy.copy(), pmf.error
    if jacobian and kT:
        g = g + 2.0 * kT * np.log(np.maximum(xi, 1e-6))

    full_range = [round(float(xi.min()), 3), round(float(xi.max()), 3)]
    truncated = False
    if xi_max is not None and xi_max < xi.max():
        keep = xi <= xi_max
        if keep.sum() < 5:
            return _no_estimate(full_range, plateau_width, jacobian, xi_max,
                                "connected stretch too short to hold a profile")
        # the bootstrap errors are indexed by the same grid, so they have to
        # follow the cut or every index below lands in the wrong place
        xi, g = xi[keep], g[keep]
        if err_of_xi is not None and err_of_xi.size == keep.size:
            err_of_xi = err_of_xi[keep]
        truncated = True

    bound_i = int(np.argmin(g))
    mask = xi >= xi.max() - plateau_width
    if mask.sum() < 3:
        mask = np.zeros_like(xi, dtype=bool)
        mask[-3:] = True

    # A plateau only means something if the profile has stopped climbing before
    # it. Compare the last plateau_width with the one before it: on a real
    # plateau the two means agree to within the noise, while a cut that landed
    # on the rising part leaves a step of several kcal/mol. Without that test a
    # truncated profile would report the height of a slope as a well depth.
    before = (xi >= xi.max() - 2 * plateau_width) & ~mask
    plateau = float(np.mean(g[mask]))
    plateau_sd = float(np.std(g[mask]))
    if before.sum() < 3:
        return _no_estimate(full_range, plateau_width, jacobian, xi_max,
                            "connected stretch is shorter than two plateau widths")
    rise = plateau - float(np.mean(g[before]))
    if rise > max(1.0, 2 * plateau_sd):
        return _no_estimate(full_range, plateau_width, jacobian, xi_max,
                            f"still rising at the end of the connected stretch: the last "
                            f"{plateau_width:g} nm sit {rise:.1f} kcal/mol above the "
                            f"{plateau_width:g} nm before them")

    dG = float(g[bound_i] - plateau)

    err = None
    if err_of_xi is not None and err_of_xi.size == xi.size:
        err = float(np.sqrt(err_of_xi[bound_i] ** 2 + np.mean(err_of_xi[mask]) ** 2))

    barrier = float(g[bound_i:].max() - g[bound_i]) if bound_i < len(g) - 1 else 0.0

    return {
        "dG": round(dG, 3),
        "dG_bootstrap_error": round(err, 3) if err is not None else None,
        "bound_xi_nm": round(float(xi[bound_i]), 3),
        "plateau_energy": round(plateau, 3),
        "plateau_roughness": round(plateau_sd, 3),
        "plateau_rise": round(rise, 3),
        "plateau_width_nm": round(plateau_width, 3),
        "unbinding_barrier": round(barrier, 3),
        "xi_range_nm": [round(float(xi.min()), 3), round(float(xi.max()), 3)],
        "xi_range_full_nm": full_range,
        "xi_connected_nm": round(float(xi_max), 3) if xi_max is not None else None,
        "truncated_at_gap": truncated,
        "no_estimate_reason": None,
        "jacobian_corrected": bool(jacobian),
    }


def _no_estimate(full_range: list[float], plateau_width: float, jacobian: bool,
                 xi_max: float | None, why: str) -> dict:
    return {
        "dG": None,
        "dG_bootstrap_error": None,
        "bound_xi_nm": None,
        "plateau_energy": None,
        "plateau_roughness": None,
        "plateau_rise": None,
        "plateau_width_nm": round(plateau_width, 3),
        "unbinding_barrier": None,
        "xi_range_nm": full_range,
        "xi_range_full_nm": full_range,
        "xi_connected_nm": round(float(xi_max), 3) if xi_max is not None else None,
        "truncated_at_gap": xi_max is not None,
        "no_estimate_reason": why,
        "jacobian_corrected": bool(jacobian),
    }


def histogram_overlap(centers: np.ndarray, hist: np.ndarray) -> dict:
    """Overlap coefficient between neighbouring windows, ordered by mean position."""
    if hist.size == 0:
        return {"overlaps": [], "min": None, "median": None, "n_gaps": 0}

    counts = np.nan_to_num(hist)
    totals = counts.sum(axis=0)
    keep = totals > 0
    counts, totals = counts[:, keep], totals[keep]
    means = (centers[:, None] * counts).sum(axis=0) / totals
    order = np.argsort(means)
    p = counts[:, order] / totals[order]

    overlaps = [float(np.minimum(p[:, i], p[:, i + 1]).sum()) for i in range(p.shape[1] - 1)]
    return {
        "overlaps": [round(o, 4) for o in overlaps],
        "window_centers_nm": [round(float(m), 3) for m in means[order]],
        "min": round(min(overlaps), 4) if overlaps else None,
        "median": round(float(np.median(overlaps)), 4) if overlaps else None,
        "mean": round(float(np.mean(overlaps)), 4) if overlaps else None,
        "counts_per_window": [int(t) for t in totals[order]],
    }


def convergence(workdir: str | Path, proto: Protocol, n_blocks: int = 4,
                xi_max: float | None = None) -> dict:
    """dG as a function of the amount of umbrella data used (cumulative blocks).

    `xi_max` is the same cut the headline estimate uses, so the drift measures
    sampling convergence rather than how the arbitrary offset across a broken
    pair happened to land in each half. A block whose shortened data no longer
    reaches a plateau reports no number instead of a spurious one.
    """
    total = proto.umbrella.time_ns * 1000
    start = proto.umbrella.discard_ns * 1000
    usable = total - start
    if usable <= 0:
        return {"blocks": []}

    blocks = []
    for k in range(1, n_blocks + 1):
        end = start + usable * k / n_blocks
        paths = run_wham(workdir, proto, prefix=f"conv{k}", begin_ps=start,
                         end_ps=end, bootstraps=0)
        pmf = load_pmf(paths)
        res = binding_free_energy(pmf, proto.analysis.plateau_width, xi_max=xi_max)
        blocks.append({"used_ns": round((end - start) / 1000, 2), "dG": res["dG"]})

    halves = []
    mid = start + usable / 2
    for tag, (b, e) in {"first_half": (start, mid), "second_half": (mid, total)}.items():
        paths = run_wham(workdir, proto, prefix=tag, begin_ps=b, end_ps=e, bootstraps=0)
        res = binding_free_energy(load_pmf(paths), proto.analysis.plateau_width, xi_max=xi_max)
        halves.append({"half": tag, "dG": res["dG"], "reason": res["no_estimate_reason"]})

    scored = [h["dG"] for h in halves if h["dG"] is not None]
    drift = abs(scored[0] - scored[1]) if len(scored) == 2 else None
    return {
        "blocks": blocks,
        "halves": halves,
        "half_split_drift": round(drift, 3) if drift is not None else None,
        "blocks_scored": sum(1 for b in blocks if b["dG"] is not None),
    }


def save(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=_encode))


def _encode(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(type(o))
