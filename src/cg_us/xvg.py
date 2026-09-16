"""Minimal GROMACS .xvg reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_xvg(path: str | Path) -> tuple[np.ndarray, dict]:
    meta: dict = {"legends": []}
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("@"):
            if " legend " in line and line.split()[1].startswith("s"):
                meta["legends"].append(line.split('"')[1] if '"' in line else "")
            elif "xaxis" in line and '"' in line:
                meta["xlabel"] = line.split('"')[1]
            elif "yaxis" in line and '"' in line:
                meta["ylabel"] = line.split('"')[1]
            elif "title" in line and '"' in line:
                meta["title"] = line.split('"')[1]
            continue
        try:
            rows.append([float(x) for x in line.split()])
        except ValueError:
            continue
    if not rows:
        return np.empty((0, 0)), meta
    width = max(len(r) for r in rows)
    data = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        data[i, :len(r)] = r
    return data, meta


def write_xvg(path: str | Path, data: np.ndarray, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    head = [f'@    title "{title}"', f'@    xaxis  label "{xlabel}"', f'@    yaxis  label "{ylabel}"', "@TYPE xy"]
    body = ["\t".join(f"{v:.6f}" for v in row) for row in data]
    Path(path).write_text("\n".join(head + body) + "\n")
