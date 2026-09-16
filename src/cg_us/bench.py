"""Measure mdrun throughput for this system before committing GPU-weeks to it.

Umbrella windows are small systems run many times over, so the two questions
that decide the wall-clock budget are which GPU offload combination is fastest
and how many windows should share one GPU. Both are hardware-specific; this
module answers them by running short mdruns instead of guessing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PERF_RE = re.compile(r"^Performance:\s+([0-9.]+)\s+([0-9.]+)", re.M)
UNSUPPORTED_RE = re.compile(
    r"(?i)(not supported|cannot be used|incompatible|unknown command-line option|"
    r"requires.*GPU|is not compatible)"
)


@dataclass
class Config:
    name: str
    flags: list[str]
    note: str = ""


DEFAULT_CONFIGS = [
    Config("nb-gpu", ["-nb", "gpu", "-pme", "cpu"], "what CHAPERONg runs by default"),
    Config("nb+pme", ["-nb", "gpu", "-pme", "gpu", "-pmefft", "gpu"], "PME off the CPU"),
    Config("nb+pme+bonded", ["-nb", "gpu", "-pme", "gpu", "-pmefft", "gpu", "-bonded", "gpu"], ""),
    Config("gpu-resident", ["-nb", "gpu", "-pme", "gpu", "-pmefft", "gpu",
                            "-bonded", "gpu", "-update", "gpu"],
           "keeps the trajectory on the GPU between steps"),
]


@dataclass
class Result:
    config: str
    ntomp: int
    concurrency: int
    ns_per_day: float | None
    aggregate_ns_per_day: float | None
    seconds: float
    ok: bool
    message: str = ""
    flags: list[str] = field(default_factory=list)


def _parse_performance(log: Path) -> float | None:
    if not log.exists():
        return None
    m = PERF_RE.search(log.read_text(errors="ignore"))
    return float(m.group(1)) if m else None


def _pick_tpr(workdir: Path) -> Path:
    for pattern in ("umbrella_win*.tpr", "pull.tpr", "npt.tpr"):
        hits = sorted(workdir.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"no .tpr to benchmark in {workdir}. Run at least up to the steered MD "
        "(cg-us run ... ) or point --workdir at a replica that has one."
    )


def _mdrun(gmx: str, tpr: Path, outdir: Path, flags: list[str], ntomp: int,
           steps: int, gpu_id: str = "", slot: int | None = None) -> tuple[float | None, str, float]:
    outdir.mkdir(parents=True, exist_ok=True)
    deffnm = outdir / "bench"
    cmd = [gmx, "mdrun", "-s", str(tpr), "-deffnm", str(deffnm),
           "-nsteps", str(steps), "-resethway", "-noconfout",
           "-ntmpi", "1", "-ntomp", str(ntomp), *flags]
    if slot is not None:
        cmd += ["-pin", "on", "-pinoffset", str(slot * ntomp), "-pinstride", "1"]
    if gpu_id:
        cmd += ["-gpu_id", gpu_id]

    started = time.time()
    proc = subprocess.run(cmd, cwd=outdir, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    log = Path(f"{deffnm}.log")
    perf = _parse_performance(log)

    if proc.returncode != 0 or perf is None:
        tail = "\n".join((proc.stdout or "").splitlines()[-6:])
        reason = "unsupported on this build" if UNSUPPORTED_RE.search(proc.stdout or "") else "failed"
        return None, f"{reason}: {tail[:300]}", elapsed
    return perf, "", elapsed


def core_sweep(cores: int) -> tuple[int, ...]:
    """A short ntomp ladder: full GPU offload usually peaks well below all cores."""
    candidates = {4, 8, max(1, cores // 2), cores}
    return tuple(sorted(c for c in candidates if 1 <= c <= cores))


def benchmark(workdir: str | Path, gmx: str = "gmx", steps: int = 10000,
              ntomp_values: tuple[int, ...] = (8,), concurrency: tuple[int, ...] = (1,),
              configs: list[Config] | None = None, gpu_id: str = "",
              keep: bool = False, cores: int | None = None) -> pd.DataFrame:
    workdir = Path(workdir)
    cores = cores or os.cpu_count() or 1
    tpr = _pick_tpr(workdir)
    configs = configs or DEFAULT_CONFIGS
    benchdir = workdir / "bench"
    benchdir.mkdir(exist_ok=True)

    results: list[Result] = []
    best_flags: list[str] | None = None

    for cfg in configs:
        for ntomp in ntomp_values:
            tag = f"{cfg.name}_omp{ntomp}"
            perf, msg, secs = _mdrun(gmx, tpr, benchdir / tag, cfg.flags, ntomp, steps, gpu_id)
            results.append(Result(cfg.name, ntomp, 1, perf, perf, secs, perf is not None, msg,
                                  list(cfg.flags)))
            print(f"  {tag:<28} " + (f"{perf:8.1f} ns/day" if perf else f"-- {msg.splitlines()[0][:60]}"))

    ok = [r for r in results if r.ok]
    if ok:
        best = max(ok, key=lambda r: r.ns_per_day)
        best_flags = best.flags
        best_ntomp = best.ntomp

    for k in concurrency:
        if k <= 1 or not best_flags:
            continue
        omp = max(1, min(best_ntomp, cores // k))
        with ThreadPoolExecutor(max_workers=k) as pool:
            runs = [pool.submit(_mdrun, gmx, tpr, benchdir / f"conc{k}_{i}",
                                best_flags, omp, steps, gpu_id, i) for i in range(k)]
            got = [r.result() for r in runs]
        perfs = [p for p, _, _ in got if p]
        if not perfs:
            continue
        each, total = sum(perfs) / len(perfs), sum(perfs)
        results.append(Result(f"{best.config} x{k}", omp, k, each, total,
                              max(s for _, _, s in got), True, "", best_flags))
        print(f"  {best.config} x{k:<21} {each:8.1f} ns/day each -> {total:8.1f} aggregate")

    if not keep:
        shutil.rmtree(benchdir, ignore_errors=True)

    df = pd.DataFrame([r.__dict__ for r in results])
    return df.drop(columns=["flags"])


def recommend(df: pd.DataFrame, windows: int, time_ns: float, replicas: int) -> dict:
    ok = df[df["ok"]]
    if ok.empty:
        return {}
    baseline = ok[ok["config"] == "nb-gpu"]["aggregate_ns_per_day"].max()
    best = ok.loc[ok["aggregate_ns_per_day"].idxmax()]
    total_ns = windows * time_ns * replicas
    out = {
        "total_ns_per_system": total_ns,
        "best_config": best["config"],
        "best_aggregate_ns_per_day": round(float(best["aggregate_ns_per_day"]), 1),
        "hours_per_system": round(24 * total_ns / float(best["aggregate_ns_per_day"]), 1),
    }
    if baseline and baseline > 0:
        out["speedup_vs_nb_gpu"] = round(float(best["aggregate_ns_per_day"]) / baseline, 2)
        out["baseline_hours_per_system"] = round(24 * total_ns / baseline, 1)
    return out
