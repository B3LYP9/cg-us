"""Fit and check points read from analysis/systems.csv files."""

import numpy as np
import pandas as pd
import pytest

from cg_us import calibration as C
from cg_us import cli


def write_root(path, rows):
    (path / "analysis").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "analysis" / "systems.csv", index=False)


def bench_rows():
    a, b = 0.25, -3.5
    rows = [dict(system=f"s{i}", role="design", n_replicas=3, dG_mean=c, dG_sd=1.0, dg_exp=a * c + b)
            for i, c in enumerate([-24, -20, -16, -12, -8, -6])]
    rows.append(dict(system="ref1", role="reference", n_replicas=3, dG_mean=-15.0, dG_sd=2.0, dg_exp=-10.0))
    rows.append(dict(system="neg1", role="negative_control", n_replicas=3, dG_mean=-6.0, dG_sd=1.0, dg_exp=np.nan))
    rows.append(dict(system="neg2", role="negative_control", n_replicas=3, dG_mean=-2.0, dG_sd=1.0, dg_exp=np.nan))
    rows.append(dict(system="no_dg", role="benchmark", n_replicas=0, dG_mean=np.nan, dG_sd=np.nan, dg_exp=-8.0))
    return rows


def test_fit_recovers_the_line_and_leaves_references_and_negatives_out(tmp_path):
    write_root(tmp_path / "bench", bench_rows())
    f = C.fit(C.read_systems(tmp_path / "bench"))
    assert f["a"] == pytest.approx(0.25) and f["b"] == pytest.approx(-3.5)
    assert f["n"] == 6 and "ref1" not in f["systems"] and "neg1" not in f["systems"]
    assert f["rmse"] == pytest.approx(0, abs=1e-9) and f["loo_rmse"] == pytest.approx(0, abs=1e-9)


def test_check_points_are_the_references_and_the_systems_of_the_check_roots(tmp_path):
    write_root(tmp_path / "bench", bench_rows())
    write_root(tmp_path / "df3", [dict(system="gdf8_df3", role="benchmark", n_replicas=3, dG_mean=-20.0, dG_sd=1.0, dg_exp=-9.5),
                                  dict(system="gdf8_s7", role="benchmark", n_replicas=2, dG_mean=-22.0, dG_sd=2.0, dg_exp=np.nan),
                                  dict(system="ctrl_neg", role="negative_control", n_replicas=3, dG_mean=-5.0, dG_sd=1.0, dg_exp=-3.0)])
    bench = C.read_systems(tmp_path / "bench")
    f = C.fit(bench)
    t = C.check_table(bench, [tmp_path / "df3"], f).set_index("system")
    assert set(t.index) == {"ref1", "gdf8_df3"}               # no dG_exp -> no point; negative control -> no point
    assert t.loc["gdf8_df3", "dG_calib"] == pytest.approx(-8.5)
    assert t.loc["gdf8_df3", "error"] == pytest.approx(1.0)
    assert t.loc["ref1", "set"] == "reference" and t.loc["gdf8_df3", "set"] == "df3"
    assert C.negative_band(bench, f) == pytest.approx((-5.0, -4.0, 2)) or C.negative_band(bench, f)[2] == 2


def test_calplot_command_writes_figure_and_table(tmp_path, capsys):
    write_root(tmp_path / "bench", bench_rows())
    write_root(tmp_path / "arm", [dict(system="prodarm_hollow", role="benchmark", n_replicas=3, dG_mean=-39.0,
                                       dG_sd=15.0, dg_exp=-11.9)])
    out = tmp_path / "rep" / "check.png"
    rc = cli.main(["calplot", "--bench", str(tmp_path / "bench"), "--check", str(tmp_path / "arm"), "--out", str(out)])
    assert rc == 0 and out.stat().st_size > 5000
    table = pd.read_csv(out.with_suffix(".csv"))
    assert "prodarm_hollow" in set(table.system)
    assert "prodarm_hollow" in capsys.readouterr().out


def test_missing_analysis_gives_a_clear_error(tmp_path, capsys):
    rc = cli.main(["calplot", "--bench", str(tmp_path / "nothing")])
    assert rc == 1 and "cg-us analyze" in capsys.readouterr().err
