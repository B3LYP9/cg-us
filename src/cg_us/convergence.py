"""Per-window sampling diagnostics.

A uniform simulation time per window is the wrong allocation: windows in the
solvated tail decorrelate in picoseconds, while windows in the desolvation
region can stay correlated for hundreds. What matters is the effective sample
size of the biased coordinate, so it is measured per window and time is spent
only where it is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .xvg import read_xvg


def pull_series(path: str | Path, discard_ps: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    data, _ = read_xvg(path)
    if data.size == 0:
        return np.empty(0), np.empty(0)
    t, xi = data[:, 0], data[:, 1]
    keep = t >= (t[0] + discard_ps)
    return t[keep], xi[keep]


def statistical_inefficiency(x: np.ndarray, cutoff: float = 6.0) -> float:
    """g = 1 + 2*sum(rho), truncated by Sokal's automatic window.

    Returns the number of samples one independent observation is worth.
    """
    n = len(x)
    if n < 16:
        return float(n)
    y = x - x.mean()
    if not np.any(y):
        return float(n)

    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, size)
    acf = np.fft.irfft(f * np.conj(f), size)[:n].real
    if acf[0] <= 0:
        return float(n)
    acf /= acf[0]

    g = 1.0
    for m in range(1, n):
        g = 1.0 + 2.0 * acf[1:m + 1].sum()
        if g < 1.0:
            return 1.0
        if m >= cutoff * g:
            break
    return float(min(max(g, 1.0), n))


def is_bimodal(xi: np.ndarray, bins: int = 40, valley: float = 0.6) -> bool:
    """Heuristic: two populated peaks with a clear valley between them."""
    if len(xi) < 100:
        return False
    counts, _ = np.histogram(xi, bins=bins)
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    peaks = [i for i in range(1, len(smooth) - 1)
             if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]
             and smooth[i] > 0.15 * smooth.max()]
    if len(peaks) < 2:
        return False
    a, b = peaks[0], peaks[-1]
    return smooth[a:b + 1].min() < valley * min(smooth[a], smooth[b])


def window_diagnostics(pullx: str | Path, window: int, discard_ps: float = 0.0,
                       target: float | None = None) -> dict:
    t, xi = pull_series(pullx, discard_ps)
    if len(t) < 8:
        return {"window": window, "file": str(Path(pullx).name), "usable": False,
                "n_samples": int(len(t))}

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    g = statistical_inefficiency(xi)
    n_eff = len(xi) / g
    half = len(xi) // 2
    sd = float(xi.std(ddof=1))
    drift = abs(float(xi[:half].mean() - xi[half:].mean()))

    return {
        "window": window,
        "file": Path(pullx).name,
        "usable": True,
        "sampled_ns": round(float(t[-1] - t[0]) / 1000, 3),
        "n_samples": int(len(xi)),
        "mean_nm": round(float(xi.mean()), 4),
        "sd_nm": round(sd, 4),
        "iact_ps": round(float(g * dt / 2), 2),
        "n_eff": round(float(n_eff), 1),
        "drift_nm": round(drift, 4),
        "drift_over_sd": round(drift / sd, 3) if sd > 0 else None,
        "bimodal": bool(is_bimodal(xi)),
        "offset_from_target_nm": round(float(xi.mean() - target), 4) if target is not None else None,
    }


def verdict(diag: dict, min_neff: float, max_drift_sd: float) -> tuple[bool, str]:
    """(needs_more_sampling, reason)."""
    if not diag.get("usable"):
        return True, "no usable pull data"
    reasons = []
    if diag["n_eff"] < min_neff:
        reasons.append(f"N_eff {diag['n_eff']:.0f} < {min_neff:.0f}")
    if diag["drift_over_sd"] is not None and diag["drift_over_sd"] > max_drift_sd:
        reasons.append(f"mean drifts {diag['drift_over_sd']:.2f} sd between halves")
    if diag["bimodal"]:
        reasons.append("histogram is bimodal (slow orthogonal motion, may not converge)")
    return bool(reasons), "; ".join(reasons)


def survey(workdir: str | Path, discard_ps: float, min_neff: float,
           max_drift_sd: float) -> list[dict]:
    workdir = Path(workdir)
    listing = workdir / "pullx_files.dat"
    if listing.exists():
        files = [workdir / l.strip() for l in listing.read_text().splitlines() if l.strip()]
    else:
        files = sorted(workdir.glob("umbrella_win*_pullx.xvg"))

    out = []
    for i, f in enumerate(files):
        if not f.exists():
            out.append({"window": i, "file": f.name, "usable": False, "n_samples": 0,
                        "needs_more": True, "reason": "missing pullx.xvg"})
            continue
        d = window_diagnostics(f, i, discard_ps)
        needs, why = verdict(d, min_neff, max_drift_sd)
        out.append({**d, "needs_more": needs, "reason": why})
    return out


def budget(diagnostics: list[dict], extend_ns: float, max_ns: float) -> list[dict]:
    """Which windows to extend and by how much, respecting the per-window ceiling."""
    plan = []
    for d in diagnostics:
        if not d.get("needs_more"):
            continue
        done = d.get("sampled_ns", 0.0)
        room = max(0.0, max_ns - done)
        if room <= 0:
            plan.append({**d, "extend_ns": 0.0, "capped": True})
            continue
        plan.append({**d, "extend_ns": round(min(extend_ns, room), 3), "capped": False})
    return plan
