"""Umbrella integration: a PMF from the mean force, not from histogram overlap.

WHAM ties neighbouring windows together through the samples they share, so a
pair with no shared samples leaves the free-energy offset between the two sides
unconstrained and the profile picks up an arbitrary step. The mean biasing force
carries the same physics without that requirement: each window reports the local
gradient of the PMF at its own mean position,

    dG/dxi |<xi>  =  -<f_bias>  =  k (<xi> - xi_ref)

and the profile follows by integrating those gradients. Windows contribute
independently, so a thin or missing overlap costs interpolation accuracy across
that one step instead of an unbounded offset for everything beyond it.

Kaestner & Thiel, J. Chem. Phys. 123, 144104 (2005); errors in 124, 234106 (2006).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .convergence import is_bimodal, statistical_inefficiency
from .xvg import read_xvg

KJ_PER_KCAL = 4.184
INIT_RE = re.compile(r"^\s*pull[-_]coord1[-_]init\s*=\s*([-\d.eE+]+)", re.M)


@dataclass
class WindowForce:
    window: int
    xi_mean: float
    xi_sd: float
    grad: float
    grad_sd: float
    n_samples: int
    n_eff: float
    bimodal: bool
    k_measured: float | None = None
    xi_ref_measured: float | None = None
    xi_ref_mdp: float | None = None
    ref_residual: float | None = None


@dataclass
class ForceProfile:
    xi: np.ndarray
    energy: np.ndarray
    error: np.ndarray
    windows: list[WindowForce] = field(default_factory=list)

    @property
    def n_bimodal(self) -> int:
        return sum(1 for w in self.windows if w.bimodal)


def reference_from_mdp(path: str | Path) -> float | None:
    """pull_coord1_init out of a per-window mdp, when the reference was pinned."""
    p = Path(path)
    if not p.exists():
        return None
    m = INIT_RE.search(p.read_text())
    return float(m.group(1)) if m else None


def window_force(pullf: str | Path, pullx: str | Path, window: int,
                 discard_ps: float = 0.0, xi_ref: float | None = None,
                 end_ps: float | None = None) -> WindowForce | None:
    """Mean force and its uncertainty for one window.

    `discard_ps` and `end_ps` are both measured from the first frame of the
    window, so a block analysis asks for the same slice of wall-clock sampling
    in every window regardless of when each one happened to be written.

    The umbrella makes f an exact linear function of xi, so regressing one on
    the other recovers both the force constant and the reference position from
    the data itself. That matters for two reasons: the gradient can then be
    written as -k(<xi> - xi_ref), which does not depend on which sign
    convention the engine used when it wrote pullf.xvg, and windows from runs
    that predate the pinned reference can still be analysed - nothing outside
    these two files is needed.
    """
    fdata, _ = read_xvg(pullf)
    xdata, _ = read_xvg(pullx)
    if fdata.size == 0 or xdata.size == 0:
        return None

    n = min(len(fdata), len(xdata))
    t, f, xi = fdata[:n, 0], fdata[:n, 1], xdata[:n, 1]
    keep = t >= (t[0] + discard_ps)
    if end_ps is not None:
        keep &= t <= (t[0] + end_ps)
    f, xi = f[keep], xi[keep]
    if len(f) < 16 or np.ptp(xi) <= 0:
        return None

    slope, intercept = np.polyfit(xi, f, 1)
    if not np.isfinite(slope) or abs(slope) < 1e-6:
        return None
    k_measured = abs(float(slope))
    ref_measured = float(-intercept / slope)

    g = statistical_inefficiency(f)
    n_eff = max(1.0, len(f) / g)
    grad_kj = -k_measured * (float(xi.mean()) - ref_measured)
    sd_kj = float(f.std(ddof=1)) / np.sqrt(n_eff)

    return WindowForce(
        window=window,
        xi_mean=float(xi.mean()),
        xi_sd=float(xi.std(ddof=1)),
        grad=grad_kj / KJ_PER_KCAL,
        grad_sd=sd_kj / KJ_PER_KCAL,
        n_samples=int(len(f)),
        n_eff=float(n_eff),
        bimodal=bool(is_bimodal(xi)),
        k_measured=k_measured,
        xi_ref_measured=round(ref_measured, 5),
        xi_ref_mdp=xi_ref,
        ref_residual=None if xi_ref is None else round(ref_measured - xi_ref, 5),
    )


def integrate(windows: list[WindowForce], points: int = 400) -> ForceProfile:
    """Trapezoidal integration of the mean force over the window centres.

    Each window enters the running integral through a fixed weight, so the
    variance is the weighted sum of the per-window variances rather than a
    random walk of correlated terms.
    """
    usable = sorted((w for w in windows if np.isfinite(w.grad)), key=lambda w: w.xi_mean)
    if len(usable) < 3:
        return ForceProfile(np.empty(0), np.empty(0), np.empty(0), usable)

    nodes = np.array([w.xi_mean for w in usable])
    grad = np.array([w.grad for w in usable])
    sd = np.array([w.grad_sd for w in usable])

    weights = np.zeros((len(nodes), len(nodes)))
    for i in range(1, len(nodes)):
        dx = nodes[i] - nodes[i - 1]
        weights[i] = weights[i - 1]
        weights[i, i - 1] += 0.5 * dx
        weights[i, i] += 0.5 * dx

    energy = weights @ grad
    variance = (weights ** 2) @ (sd ** 2)

    xi = np.linspace(nodes[0], nodes[-1], points)
    return ForceProfile(
        xi=xi,
        energy=np.interp(xi, nodes, energy),
        error=np.interp(xi, nodes, np.sqrt(variance)),
        windows=usable,
    )


def profile_from_windows(workdir: str | Path, discard_ps: float, k: float | None = None,
                         listing: str = "pullx_files.dat",
                         end_ps: float | None = None) -> ForceProfile:
    """`k` is only compared against the constant recovered from the data."""
    workdir = Path(workdir)
    names = _window_files(workdir, listing)
    forces = []
    for i, pullx in enumerate(names):
        pullf = pullx.with_name(pullx.name.replace("_pullx.xvg", "_pullf.xvg"))
        if not (pullx.exists() and pullf.exists()):
            continue
        tag = pullx.name.replace("umbrella_", "").replace("_pullx.xvg", "")
        ref = reference_from_mdp(workdir / f"md_umbrella_{tag}.mdp")
        w = window_force(pullf, pullx, i, discard_ps=discard_ps, xi_ref=ref, end_ps=end_ps)
        if w is not None:
            forces.append(w)
    return integrate(forces)


def _window_files(workdir: Path, listing: str) -> list[Path]:
    f = workdir / listing
    if f.exists():
        return [workdir / line.strip() for line in f.read_text().splitlines() if line.strip()]
    return sorted(workdir.glob("umbrella_win*_pullx.xvg"))


def agreement(wham_xi: np.ndarray, wham_g: np.ndarray,
              ui: ForceProfile, xi_max: float | None = None) -> dict:
    """How far the two estimators sit apart where both are supposed to be valid.

    Both profiles are shifted to their own minimum first: WHAM and umbrella
    integration each fix the zero arbitrarily, and only the shape is comparable.
    A large residual on the connected stretch means the force-based profile
    cannot be trusted on the disconnected one either.
    """
    if wham_xi.size == 0 or ui.xi.size == 0:
        return {"rms": None, "max": None, "n_points": 0, "xi_range": None}

    lo = max(wham_xi.min(), ui.xi.min())
    hi = min(wham_xi.max(), ui.xi.max())
    if xi_max is not None:
        hi = min(hi, xi_max)
    if hi <= lo:
        return {"rms": None, "max": None, "n_points": 0, "xi_range": None}

    grid = np.linspace(lo, hi, 200)
    a = np.interp(grid, wham_xi, wham_g)
    b = np.interp(grid, ui.xi, ui.energy)
    a, b = a - a.min(), b - b.min()
    d = a - b
    return {
        "rms": round(float(np.sqrt(np.mean(d ** 2))), 3),
        "max": round(float(np.max(np.abs(d))), 3),
        "n_points": int(grid.size),
        "xi_range": [round(float(lo), 3), round(float(hi), 3)],
    }


def summarise(ui: ForceProfile) -> dict:
    if not ui.windows:
        return {"n_windows": 0}
    spacing = np.diff([w.xi_mean for w in ui.windows])
    return {
        "n_windows": len(ui.windows),
        "xi_range_nm": [round(float(ui.xi.min()), 3), round(float(ui.xi.max()), 3)],
        "max_node_spacing_nm": round(float(spacing.max()), 4) if spacing.size else None,
        "grad_sd_median": round(float(np.median([w.grad_sd for w in ui.windows])), 3),
        "n_eff_min": round(float(min(w.n_eff for w in ui.windows)), 1),
        "bimodal_windows": ui.n_bimodal,
        "k_measured_median": round(float(np.median([w.k_measured for w in ui.windows
                                                    if w.k_measured])), 1),
        "ref_residual_max_nm": (round(max(abs(w.ref_residual) for w in ui.windows
                                          if w.ref_residual is not None), 4)
                                if any(w.ref_residual is not None for w in ui.windows) else None),
    }


def window_span_ps(workdir: str | Path, listing: str = "pullx_files.dat") -> float | None:
    """How much sampling the shortest window actually holds.

    `extend` lengthens windows past the protocol's time_ns, so the block
    analysis has to ask the files rather than the configuration; otherwise an
    extended run is scored on its first few nanoseconds and looks unchanged.
    """
    spans = []
    for pullx in _window_files(Path(workdir), listing):
        if not pullx.exists():
            continue
        d, _ = read_xvg(pullx)
        if d.size:
            spans.append(float(d[-1, 0] - d[0, 0]))
    return min(spans) if spans else None


def convergence_blocks(workdir: str | Path, discard_ps: float, total_ps: float | None,
                       n_blocks: int, score, k: float | None = None) -> list[dict]:
    """dG against the amount of umbrella data used, from the mean force alone.

    The WHAM version of this test re-runs gmx wham once per block and then
    applies the connected-range cut, so on a ladder with even one broken pair
    every block reports nothing and the drift is silently unavailable - which
    is what left 89 of 96 replicas in the first campaign with no convergence
    evidence at all. The force-based profile needs no ladder and no external
    call: a block is just a shorter slice of pullf/pullx, so the whole curve
    comes out of files that are already on disk.

    `score` turns a ForceProfile into the reported dict.
    """
    if total_ps is None:
        total_ps = window_span_ps(workdir)
    if total_ps is None:
        return []
    usable = total_ps - discard_ps
    if usable <= 0 or n_blocks < 1:
        return []
    out = []
    for i in range(1, n_blocks + 1):
        end = discard_ps + usable * i / n_blocks
        prof = profile_from_windows(workdir, discard_ps, k=k, end_ps=end)
        res = score(prof) if prof.xi.size else {"dG": None}
        out.append({"used_ns": round((end - discard_ps) / 1000, 2),
                    "dG": res.get("dG"),
                    "n_eff_min": (round(min(w.n_eff for w in prof.windows), 1)
                                  if prof.windows else None)})
    return out


def block_drift(blocks: list[dict]) -> dict:
    """How much the estimate is still moving over the last half of the data.

    A profile that has stopped changing gives a flat tail; one that is still
    walking has not converged, whatever its error bar says. The slope is
    reported per nanosecond so it can be compared across runs of different
    length.
    """
    pts = [(b["used_ns"], b["dG"]) for b in blocks if b.get("dG") is not None]
    if len(pts) < 3:
        return {"drift_kcal": None, "slope_kcal_per_ns": None, "n_points": len(pts)}
    t = np.array([p[0] for p in pts])
    g = np.array([p[1] for p in pts])
    half = t >= t.max() / 2
    slope = float(np.polyfit(t[half], g[half], 1)[0]) if half.sum() >= 2 else None
    return {
        "drift_kcal": round(float(g[-1] - g[0]), 3),
        "slope_kcal_per_ns": round(slope, 3) if slope is not None else None,
        "n_points": len(pts),
    }
