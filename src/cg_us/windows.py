"""Umbrella window placement from the steered-MD distance profile."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_distance_summary(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    frames, dists = [], []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 2:
            try:
                frames.append(int(f[0]))
                dists.append(float(f[1]))
            except ValueError:
                continue
    return np.array(frames), np.array(dists)


def write_distance_summary(path: str | Path, frames: np.ndarray, dists: np.ndarray) -> None:
    Path(path).write_text("\n".join(f"{int(f)}\t{d:.4f}" for f, d in zip(frames, dists)) + "\n")


def _nearest_unused(frames: np.ndarray, dists: np.ndarray, target: float,
                    used: set[int]) -> int | None:
    """Index into `frames`/`dists` of the frame closest to `target`, skipping
    any frame number already in `used`. None if every frame is taken."""
    order = np.argsort(np.abs(dists - target))
    return next((i for i in order if int(frames[i]) not in used), None)


def select_windows(frames: np.ndarray, dists: np.ndarray, spacing: float,
                   max_distance: float, dense_until: float | None = None,
                   dense_spacing: float | None = None) -> list[dict]:
    """Pick one SMD frame per target COM distance on a (optionally graded) grid.

    Monotonicity of the SMD distance trace is not assumed: for each grid point
    the nearest sampled frame is taken, and frames are never reused.
    """
    if len(frames) == 0:
        raise ValueError("empty distance profile")

    d0 = float(dists[0])
    grid: list[float] = []
    d = d0
    while d <= d0 + max_distance + 1e-9:
        grid.append(d)
        step = dense_spacing if (dense_until and d < d0 + dense_until and dense_spacing) else spacing
        d += step

    used: set[int] = set()
    windows: list[dict] = []
    for target in grid:
        pick = _nearest_unused(frames, dists, target, used)
        if pick is None:
            continue
        used.add(int(frames[pick]))
        windows.append({
            "window": len(windows),
            "frame": int(frames[pick]),
            "target_distance": round(float(target), 4),
            "distance": round(float(dists[pick]), 4),
            "deviation": round(float(dists[pick] - target), 4),
        })
    return windows


def write_config_list(path: str | Path, windows: list[dict]) -> None:
    lines = ["frame#\tdist\td_dist"]
    prev = None
    for w in windows:
        d = w["distance"]
        lines.append(f"{w['frame']}\t{d}\t{'nil' if prev is None else round(d - prev, 3)}")
        prev = d
    Path(path).write_text("\n".join(lines) + "\n")


def read_config_list(path: str | Path) -> list[dict]:
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        f = line.split()
        if not f or f[0].startswith("frame") or "#" in f[0]:
            continue
        out.append({"window": len(out), "frame": int(f[0]), "distance": float(f[1])})
    return out


def spacing_report(windows: list[dict]) -> dict:
    d = np.array([w["distance"] for w in windows])
    gaps = np.diff(d)
    return {
        "n_windows": len(windows),
        "range_nm": [round(float(d.min()), 3), round(float(d.max()), 3)],
        "mean_spacing_nm": round(float(gaps.mean()), 4) if len(gaps) else None,
        "max_spacing_nm": round(float(gaps.max()), 4) if len(gaps) else None,
        "n_gaps_above_2x": int((gaps > 2 * np.median(gaps)).sum()) if len(gaps) else 0,
    }


def find_gaps(overlaps: list[float], centers_nm: list[float], threshold: float) -> list[dict]:
    """Adjacent window pairs whose histogram overlap is below `threshold`.

    WHAM fixes the offset between two windows from the samples they share; a
    pair with (near) no shared samples leaves that offset unconstrained no
    matter how long either window runs, because a longer run of window i or
    i+1 does not manufacture a configuration the other one would have
    visited. The fix is a window physically between them, not more time in
    either - which is exactly what `extend` cannot do and this can.

    `centers_nm` is `len(overlaps) + 1` long (one center per window, ordered by
    empirical mean position, as `wham.histogram_overlap` returns it).
    """
    if len(centers_nm) != len(overlaps) + 1:
        raise ValueError("centers_nm must have one more entry than overlaps")
    return [
        {
            "before_nm": round(centers_nm[i], 4),
            "after_nm": round(centers_nm[i + 1], 4),
            "target_distance": round((centers_nm[i] + centers_nm[i + 1]) / 2, 4),
            "overlap": overlaps[i],
        }
        for i in range(len(overlaps))
        if overlaps[i] < threshold
    ]


def select_gap_frames(frames: np.ndarray, dists: np.ndarray, gaps: list[dict],
                      used: set[int]) -> list[dict]:
    """One nearest not-yet-used SMD frame per gap midpoint.

    `used` should already contain every frame a window (old or new) has
    claimed; frames are never reused, including across the gaps in one call.
    """
    used = set(used)
    picked: list[dict] = []
    for gap in gaps:
        target = gap["target_distance"]
        pick = _nearest_unused(frames, dists, target, used)
        if pick is None:
            continue
        used.add(int(frames[pick]))
        picked.append({
            "frame": int(frames[pick]),
            "target_distance": target,
            "distance": round(float(dists[pick]), 4),
            "deviation": round(float(dists[pick] - target), 4),
            "gap_before_nm": gap["before_nm"],
            "gap_after_nm": gap["after_nm"],
            "gap_overlap": gap["overlap"],
        })
    return picked
