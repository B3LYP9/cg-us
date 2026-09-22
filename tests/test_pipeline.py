"""End-to-end exercise of prep -> analysis -> report with synthetic WHAM output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cg_us import analysis, cli, experiment, prep, wham
from cg_us.config import Protocol
from cg_us.manifest import read_manifest
from cg_us.xvg import write_xvg

BENCH = Path(__file__).resolve().parents[1] / "examples"


def synth_pmf(depth: float, n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xi = np.linspace(0.6, 3.1, n)
    g = depth * np.exp(-((xi - 0.8) / 0.35) ** 2) + rng.normal(0, 0.05, n)
    return xi, g - g[-15:].mean()


def fake_wham_factory(depth_by_system: dict, noise: float = 0.4):
    def fake(workdir, proto, prefix="wham", begin_ps=None, end_ps=None, bootstraps=None):
        workdir = Path(workdir)
        out = workdir / "analysis"
        out.mkdir(exist_ok=True)
        system = workdir.parent.name
        replica = int(workdir.name.replace("rep", ""))
        depth = depth_by_system[system] + noise * (replica - 2)
        xi, g = synth_pmf(depth, seed=replica)

        write_xvg(out / f"{prefix}_pmf.xvg", np.column_stack([xi, g]))
        err = np.full_like(xi, 0.2)
        write_xvg(out / f"{prefix}_bsres.xvg", np.column_stack([xi, g, err]))

        centers = xi
        n_win = 24
        mus = np.linspace(xi.min() + 0.05, xi.max() - 0.05, n_win)
        hist = np.exp(-0.5 * ((centers[:, None] - mus[None, :]) / 0.06) ** 2) * 4000
        write_xvg(out / f"{prefix}_hist.xvg", np.column_stack([centers, hist]))

        return {"pmf": out / f"{prefix}_pmf.xvg", "hist": out / f"{prefix}_hist.xvg",
                "bsres": out / f"{prefix}_bsres.xvg", "bsprof": None}
    return fake


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    manifest = BENCH / "manifest.csv"
    if not manifest.exists():
        pytest.skip("example manifest not available")
    root = tmp_path / "runs"
    proto = Protocol()
    proto.replicas = 3
    proto.umbrella.time_ns = 5.0
    root.mkdir(parents=True)
    proto.dump(root / "protocol.yaml")

    entries = read_manifest(manifest, BENCH)
    for e in entries:
        prep.prepare_entry(e, proto, root)
        for rep in range(1, proto.replicas + 1):
            (prep.replica_dir(root, e, rep) / "tpr_files.dat").write_text("a.tpr\n")

    (root / "run_state.json").write_text(json.dumps({
        "manifest": str(manifest), "manifest_root": str(BENCH),
        "systems": [e.name for e in entries]}))

    depths = {e.name: (e.dg_exp if e.dg_exp is not None else -3.0) for e in entries}
    monkeypatch.setattr(wham, "run_wham", fake_wham_factory(depths))
    monkeypatch.setattr(analysis.W, "run_wham", fake_wham_factory(depths))
    return root


def test_full_analysis(prepared):
    rc = cli.cmd_analyze(type("A", (), {"root": str(prepared), "no_convergence": True})())
    assert rc == 0

    report = prepared / "report.html"
    assert report.exists() and report.stat().st_size > 50_000

    results = json.loads((prepared / "analysis" / "results.json").read_text())
    assert len(results["systems"]) == 6
    cmp_ = results["comparison"]
    assert cmp_["n_in_regression"] == 5
    assert cmp_["MAE"] < 1.0
    assert cmp_["pearson_r"] > 0.9
    assert cmp_["controls"][0]["system"] == "actrIIb_bimagrumab_cdr3"

    for name in ("correlation.png", "replica_spread.png"):
        assert (prepared / "analysis" / name).stat().st_size > 5000


def test_replica_statistics_are_consistent(prepared):
    cli.cmd_analyze(type("A", (), {"root": str(prepared), "no_convergence": True})())
    import pandas as pd
    sys_df = pd.read_csv(prepared / "analysis" / "systems.csv")
    rep_df = pd.read_csv(prepared / "analysis" / "replicas.csv")
    assert len(rep_df) == 18
    for _, row in sys_df.iterrows():
        subset = rep_df[rep_df["system"] == row["system"]]["dG_kcal"]
        assert row["dG_mean"] == pytest.approx(subset.mean(), abs=0.01)
        assert row["dG_sd"] == pytest.approx(subset.std(ddof=1), abs=0.01)


def test_analyze_system_filter_reuses_unselected_systems(prepared, monkeypatch):
    """--system should only re-run wham for the requested systems; everything
    else must be pulled from its own replica_result.json (both to save the
    wham/UI cost, and so a --system-scoped run still produces a full report
    covering every system, not a truncated one)."""
    full_args = type("A", (), {"root": str(prepared), "no_convergence": True,
                                "system": None, "estimator": None})()
    assert cli.cmd_analyze(full_args) == 0

    before = json.loads((prepared / "analysis" / "results.json").read_text())
    all_systems = sorted(s["system"] for s in before["systems"])
    assert len(all_systems) == 6
    target_system = all_systems[0]

    calls = []
    real_run_wham = wham.run_wham

    def counting_run_wham(workdir, *a, **k):
        calls.append(Path(workdir).parent.name)  # the system directory name
        return real_run_wham(workdir, *a, **k)

    monkeypatch.setattr(wham, "run_wham", counting_run_wham)
    monkeypatch.setattr(analysis.W, "run_wham", counting_run_wham)

    scoped_args = type("A", (), {"root": str(prepared), "no_convergence": True,
                                 "system": [target_system], "estimator": None})()
    assert cli.cmd_analyze(scoped_args) == 0

    # only the requested system's replicas paid for a fresh wham call
    assert calls and set(calls) == {target_system}

    after = json.loads((prepared / "analysis" / "results.json").read_text())
    after_systems = sorted(s["system"] for s in after["systems"])
    assert after_systems == all_systems, "scoped analyze must not drop the other systems"

    before_by_name = {s["system"]: s for s in before["systems"]}
    after_by_name = {s["system"]: s for s in after["systems"]}
    for name in all_systems:
        if name == target_system:
            continue
        assert after_by_name[name]["dG_mean"] == before_by_name[name]["dG_mean"], (
            f"{name} was not selected but its result changed anyway")


def test_extend_fill_gaps_dry_run_reads_the_right_config_field(prepared, capsys):
    """Regression test: cmd_extend's --fill-gaps path read `proto.umbrella.overlap_min`,
    which does not exist (it lives on `proto.analysis`), and crashed with an
    AttributeError the first time anyone ran `cg-us extend --fill-gaps --dry-run`."""
    cli.cmd_analyze(type("A", (), {"root": str(prepared), "no_convergence": True})())

    args = type("A", (), {
        "root": str(prepared), "system": None, "replica": None,
        "rounds": 1, "dry_run": True, "fill_gaps": True,
    })()
    rc = cli.cmd_extend(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "gap(s) below overlap" in out
    assert "gap(s) found in total (dry run, nothing filled)" in out
    assert "ns in total" not in out  # the time-extend summary line does not apply here
