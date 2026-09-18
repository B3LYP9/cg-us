import os
import json
from pathlib import Path

import numpy as np
import pytest

from cg_us import structure, topology, windows
from cg_us.manifest import dg_from_kd


def test_dg_from_kd():
    assert dg_from_kd(5.8e-7, 298.15) == pytest.approx(-8.508, abs=0.01)
    assert dg_from_kd(2e-11, 298.15) == pytest.approx(-14.596, abs=0.01)


def _pdb(tmp_path):
    lines = []
    serial = 1
    for chain, base in (("A", 0.0), ("B", 20.0)):
        for i, (name, elem) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]):
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} ALA {chain}{1:4d}    "
                f"{base + i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {elem:>2s}"
            )
            serial += 1
    p = tmp_path / "t.pdb"
    p.write_text("\n".join(lines) + "\nEND\n")
    return p


def test_orientation_puts_binder_on_z(tmp_path):
    atoms = structure.read_pdb(_pdb(tmp_path))
    oriented, info = structure.orient_for_pull(atoms, "A", "B")
    axis = structure.com(structure.select(oriented, "B")) - structure.com(structure.select(oriented, "A"))
    assert axis[2] > 0
    assert np.allclose(axis[:2], 0, atol=1e-6)
    assert info["com_distance_nm"] == pytest.approx(2.0, abs=1e-6)


def _cyclic_pdb(tmp_path, head_tail_distance):
    """Chain B: two residues whose N(first) and C(last) sit at a chosen distance."""
    rows = [
        ("N", 0.0), ("CA", 1.5), ("C", 2.5),
    ]
    lines = []
    serial = 1
    for i, (name, x) in enumerate(rows):
        lines.append(
            f"ATOM  {serial:5d}  {name:<3s} ALA B{1:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
        serial += 1
    for name, x in (("N", 10.0), ("CA", 11.0), ("C", head_tail_distance)):
        lines.append(
            f"ATOM  {serial:5d}  {name:<3s} GLY B{2:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
        serial += 1
    p = tmp_path / f"c{head_tail_distance}.pdb"
    p.write_text("\n".join(lines) + "\nEND\n")
    return p


@pytest.mark.parametrize("distance,expected", [(1.33, True), (3.5, True), (4.5, False), (0.2, False)])
def test_cyclisation_follows_the_pdb2gmx_window(tmp_path, distance, expected):
    atoms = structure.read_pdb(_cyclic_pdb(tmp_path, distance))
    rep = structure.describe_chain(atoms, "B", cyclic_min=0.5, cyclic_max=4.0)
    assert rep.head_tail_distance == pytest.approx(distance, abs=1e-3)
    assert rep.cyclic is expected


@pytest.mark.parametrize("cyclic_type,expected", [
    ("head-to-tail backbone + Cys3-Cys11 (bicyclic)", 0.4),
    ("head to tail", 0.4),
    ("", 0.4),
    ("linear", 0.25),
    ("disulfide-cyclised CDR3 graft", 0.25),
])
def test_effective_lb_respects_the_manifest(cyclic_type, expected):
    from cg_us.config import Protocol
    from cg_us.manifest import Entry
    from cg_us.prep import effective_lb

    proto = Protocol()
    entry = Entry(name="x", pdb=Path("x.pdb"), target="A", binder="B", cyclic_type=cyclic_type)
    assert effective_lb(entry, proto) == expected

    proto.prep.cyclic_lb_from_manifest = False
    assert effective_lb(entry, proto) == 0.4


def test_gmx_shim_injects_cyclic_thresholds(tmp_path):
    import subprocess
    from cg_us.config import Protocol
    from cg_us.prep import write_gmx_shim

    fake = tmp_path / "fake_gmx"
    fake.write_text('#!/bin/bash\necho "ARGS: $*"\ncat\n')
    fake.chmod(0o755)

    proto = Protocol()
    proto.run.gmx = str(fake)
    proto.prep.cyclic_lb = 0.4
    shim = write_gmx_shim(tmp_path, proto)

    out = subprocess.run([str(shim), "pdb2gmx", "-f", "x.pdb"], input="1\n",
                         text=True, capture_output=True)
    assert "-sb 0.05 -lb 0.4" in out.stdout
    passthrough = subprocess.run([str(shim), "grompp", "-f", "x.mdp"], input="",
                                 text=True, capture_output=True)
    assert "-lb" not in passthrough.stdout


def test_gmx_shim_retries_without_hidden_options(tmp_path):
    import subprocess
    from cg_us.config import Protocol
    from cg_us.prep import write_gmx_shim

    fake = tmp_path / "picky_gmx"
    fake.write_text(
        '#!/bin/bash\n'
        'for a in "$@"; do if [ "$a" = "-lb" ]; then\n'
        '  echo "Unknown command-line option -lb" >&2; exit 1; fi; done\n'
        'echo "ok: $*"\ncat > /dev/null\n'
    )
    fake.chmod(0o755)

    proto = Protocol()
    proto.run.gmx = str(fake)
    shim = write_gmx_shim(tmp_path, proto)

    out = subprocess.run([str(shim), "pdb2gmx", "-f", "x.pdb"], input="1\n",
                         text=True, capture_output=True)
    assert out.returncode == 0
    assert "ok: pdb2gmx -f x.pdb" in out.stdout
    assert "retrying pdb2gmx without them" in out.stderr


def test_pull_group_indices(tmp_path):
    top = tmp_path / "topol.top"
    top.write_text(
        "[ moleculetype ]\nProtein_chain_A     3\n\n[ atoms ]\n"
        "     1   NH3  1  ALA  N  1  -0.3  14.0\n"
        "     2    CT  1  ALA CA  2   0.1  12.0\n\n"
        "[ moleculetype ]\nProtein_chain_B     3\n\n[ atoms ]\n"
        "     1   NH3  1  GLY  N  1  -0.3  14.0\n\n"
        "[ system ]\ntest\n\n[ molecules ]\nProtein_chain_A 1\nProtein_chain_B 1\nSOL 10\n"
    )
    groups = topology.pull_group_indices(top, "A", "B")
    assert groups["Target"] == [1, 2]
    assert groups["Binder"] == [3]


def test_window_selection_is_monotonic_and_unique():
    frames = np.arange(500)
    dists = 0.5 + np.linspace(0, 2.5, 500) + np.random.default_rng(0).normal(0, 0.01, 500)
    picked = windows.select_windows(frames, dists, spacing=0.1, max_distance=2.0)
    chosen = [w["frame"] for w in picked]
    assert len(chosen) == len(set(chosen))
    assert max(abs(w["deviation"]) for w in picked) < 0.1
    rep = windows.spacing_report(picked)
    assert rep["n_windows"] == len(picked)


def test_find_gaps_flags_only_pairs_below_threshold():
    # 5 windows -> 4 neighbour pairs; only the middle one is broken.
    overlaps = [0.15, 0.20, 0.0008, 0.12]
    centers = [0.5, 0.9, 1.3, 1.7, 2.1]
    gaps = windows.find_gaps(overlaps, centers, threshold=0.03)
    assert len(gaps) == 1
    g = gaps[0]
    assert g["before_nm"] == 1.3
    assert g["after_nm"] == 1.7
    assert g["target_distance"] == pytest.approx(1.5)
    assert g["overlap"] == 0.0008


def test_find_gaps_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one more entry"):
        windows.find_gaps([0.1, 0.1], [0.5, 0.9], threshold=0.03)


def test_select_gap_frames_picks_nearest_unused_frame_per_gap():
    frames = np.arange(20)
    dists = np.linspace(0.5, 2.5, 20)  # 0.5, 0.605, 0.711, ...
    gaps = [
        {"before_nm": 0.9, "after_nm": 1.3, "target_distance": 1.1, "overlap": 0.001},
        {"before_nm": 1.9, "after_nm": 2.3, "target_distance": 2.1, "overlap": 0.0005},
    ]
    picked = windows.select_gap_frames(frames, dists, gaps, used=set())
    assert len(picked) == 2
    for p, gap in zip(picked, gaps):
        assert abs(p["distance"] - gap["target_distance"]) <= (dists[1] - dists[0])
        assert p["gap_before_nm"] == gap["before_nm"]
        assert p["gap_overlap"] == gap["overlap"]
    assert picked[0]["frame"] != picked[1]["frame"]


def test_select_gap_frames_skips_frames_already_used():
    frames = np.array([0, 1, 2])
    dists = np.array([1.0, 1.05, 1.1])
    gap = {"before_nm": 0.9, "after_nm": 1.2, "target_distance": 1.0, "overlap": 0.0}
    picked = windows.select_gap_frames(frames, dists, [gap], used={0})
    assert picked[0]["frame"] == 1  # frame 0 is the exact match but already used

    picked_none = windows.select_gap_frames(frames, dists, [gap], used={0, 1, 2})
    assert picked_none == []


def _atoms(n, start_res=1):
    names = ["N", "CA", "C", "O", "HA"]
    return "\n".join(
        f"  {i + 1} CT {start_res + i // 5} ALA {names[i % 5]} {i + 1} 0.0 12.0"
        for i in range(n)
    )


def _split_topology(tmp_path):
    """pdb2gmx layout for multi-chain input: moleculetypes live in included itps."""
    (tmp_path / "topol_Protein_chain_B.itp").write_text(
        "[ moleculetype ]\n; Name nrexcl\nProtein_chain_B     3\n\n[ atoms ]\n"
        + _atoms(10)
        + '\n\n; Include Position restraint file\n#ifdef POSRES\n'
          '#include "posre_Protein_chain_B.itp"\n#endif\n'
    )
    (tmp_path / "topol_Protein_chain_A.itp").write_text(
        "[ moleculetype ]\n; Name nrexcl\nProtein_chain_A     3\n\n[ atoms ]\n"
        + _atoms(5)
        + '\n\n#ifdef POSRES\n#include "posre_Protein_chain_A.itp"\n#endif\n'
    )
    (tmp_path / "posre_Protein_chain_B.itp").write_text("[ position_restraints ]\n 1 1 1000 1000 1000\n")
    top = tmp_path / "topol.top"
    top.write_text(
        '#include "./charmm36-jul2022.ff/forcefield.itp"\n\n'
        '; Include chain topologies\n'
        '#include "topol_Protein_chain_B.itp"\n'
        '#include "topol_Protein_chain_A.itp"\n\n'
        '#include "./charmm36-jul2022.ff/tip3p.itp"\n\n'
        "[ system ]\ntest\n\n[ molecules ]\n"
        "Protein_chain_B 1\nProtein_chain_A 1\nSOL 100\n"
    )
    return top


def test_parser_follows_chain_includes(tmp_path):
    top = _split_topology(tmp_path)
    moltypes, molecules = topology.parse_topology(top)
    assert [m.name for m in moltypes] == ["Protein_chain_B", "Protein_chain_A"]
    assert [m.n_atoms for m in moltypes] == [10, 5]
    assert molecules[0] == ("Protein_chain_B", 1)
    assert topology.chain_moltypes(top, "B", "A") == {
        "Target": "Protein_chain_B", "Binder": "Protein_chain_A"}

    groups = topology.pull_group_indices(top, "B", "A")
    assert groups["Target"] == list(range(1, 11))
    assert groups["Binder"] == list(range(11, 16))


def test_restraints_land_in_the_chain_itp(tmp_path):
    top = _split_topology(tmp_path)
    topology.restraint_itp(top, "Protein_chain_B", tmp_path / "posre_target.itp")
    where = topology.wire_restraints(top, "Protein_chain_B", "posre_target.itp", "POSRES_TARGET")

    assert where.name == "topol_Protein_chain_B.itp"
    assert "POSRES_TARGET" in where.read_text()
    assert "POSRES_TARGET" not in top.read_text()
    assert "POSRES_TARGET" not in (tmp_path / "topol_Protein_chain_A.itp").read_text()
    before = where.read_text()
    topology.wire_restraints(top, "Protein_chain_B", "posre_target.itp", "POSRES_TARGET")
    assert where.read_text() == before


def test_chain_mapping_falls_back_to_order(tmp_path):
    """Chain letters lost (merged/renamed): target is the first protein molecule."""
    (tmp_path / "chainless.itp").write_text(
        "[ moleculetype ]\nProtein     3\n\n[ atoms ]\n" + _atoms(10) + "\n\n"
        "[ moleculetype ]\nProtein2     3\n\n[ atoms ]\n" + _atoms(5) + "\n"
    )
    top = tmp_path / "topol.top"
    top.write_text('#include "chainless.itp"\n\n[ system ]\nx\n\n[ molecules ]\n'
                   "Protein 1\nProtein2 1\n")
    assert topology.chain_moltypes(top, "A", "B") == {"Target": "Protein", "Binder": "Protein2"}
    groups = topology.pull_group_indices(top, "A", "B")
    assert groups["Binder"] == list(range(11, 16))


def test_restraint_itp_backbone_only(tmp_path):
    top = tmp_path / "topol.top"
    top.write_text(
        "[ moleculetype ]\nProtein_chain_A 3\n\n[ atoms ]\n"
        "  1 NH3 1 ALA  N 1 -0.3 14.0\n"
        "  2  CT 1 ALA CA 2  0.1 12.0\n"
        "  3  HA 1 ALA HA 3  0.1  1.0\n"
        "  4   C 1 ALA  C 4  0.5 12.0\n"
        "  5   O 1 ALA  O 5 -0.5 16.0\n\n[ system ]\nx\n\n[ molecules ]\nProtein_chain_A 1\n"
    )
    out = topology.restraint_itp(top, "Protein_chain_A", tmp_path / "posre_target.itp")
    body = [l for l in out.read_text().splitlines() if l and not l.startswith((";", "["))]
    assert [int(l.split()[0]) for l in body] == [1, 2, 4, 5]

    topology.wire_restraints(top, "Protein_chain_A", "posre_target.itp", "POSRES_TARGET")
    text = top.read_text()
    assert "#ifdef POSRES_TARGET" in text
    assert text.index("POSRES_TARGET") < text.index("[ system ]")


def _fake_chaperong(tmp_path, executable=True):
    modules = tmp_path / "CHAP_modules"
    modules.mkdir(parents=True, exist_ok=True)
    for m in ("CHAP_deffxn.sh", "CHAP_ana.sh", "CHAP_sim.sh", "CHAP_colPar.sh"):
        (modules / m).write_text("#\n")
    launcher = modules / "run_CHAPERONg.sh"
    launcher.write_text("#!/bin/bash\necho hi\n")
    if executable:
        launcher.chmod(0o755)
    return launcher


def test_locate_prefers_absolute_path(tmp_path):
    from cg_us.backends import chaperong

    launcher = _fake_chaperong(tmp_path)
    script_path, root = chaperong.locate(str(launcher))
    assert script_path == launcher.resolve()
    assert root == tmp_path.resolve()


def test_locate_falls_back_to_chaperong_path(tmp_path, monkeypatch):
    from cg_us.backends import chaperong

    _fake_chaperong(tmp_path)
    monkeypatch.setenv("CHAPERONg_PATH", str(tmp_path))
    monkeypatch.setattr(chaperong.shutil, "which", lambda *_: None)
    script_path, root = chaperong.locate("run_CHAPERONg.sh")
    assert script_path.name == "run_CHAPERONg.sh"
    assert root == tmp_path.resolve()


def test_locate_reports_where_it_looked(monkeypatch):
    from cg_us.backends import chaperong

    monkeypatch.delenv("CHAPERONg_PATH", raising=False)
    monkeypatch.setattr(chaperong.shutil, "which", lambda *_: None)
    with pytest.raises(RuntimeError, match="cannot find the CHAPERONg launcher"):
        chaperong.locate("run_CHAPERONg.sh")


def test_preflight_rejects_incomplete_install(tmp_path):
    from cg_us.backends import chaperong
    from cg_us.config import Protocol

    launcher = _fake_chaperong(tmp_path)
    (tmp_path / "CHAP_modules" / "CHAP_sim.sh").unlink()
    proto = Protocol()
    proto.run.chaperong = str(launcher)
    with pytest.raises(RuntimeError, match="CHAP_sim.sh"):
        chaperong.preflight(proto)


FAKE_CHAPERONG_STAGE0 = r'''#!/bin/bash
read -p " *ENTER YOUR CHOICE HERE (1 or 2): " a
read -p " *Enter 1 or 2: " b
read -p " Initiation stage: " stage
if [ "$stage" = "14" ]; then
  echo "stage14: extracting frames"
  printf 'umbrella_win0_conf0.tpr\numbrella_win1_conf7.tpr\n' > tpr_files.dat
  exit 0
fi
touch topol.top solv_ions.gro
read -p " Do you want to proceed? (yes/no): " c
read -p "ENTER A RESPONSE HERE (1 or 2): " d
read -p "Center dimensions (x y z):" e
read -p "Box dimensions (x y z):" f
read -p "ENTER A RESPONSE HERE (1 or 2): " g
read -p "Do you want to run an optional NVT equilibration? (yes/no): " h
read -p "Do you need to make custom index for pulling groups? (yes/no): " i
echo "$i" > index_answer.txt
touch pull.gro pull.xtc
echo "  Do you want to proceed to making a movie summarized into 200-300 frames?"
read -p "  Enter 1 or 2 here: " j
echo "MOVIE WAS RENDERED" > movie_marker.txt
read -p " Do you want to proceed with umbrella sampling? (yes/no): " k
printf 'umbrella_win0_conf0.tpr\n' > tpr_files.dat
'''


def _stub_replica(tmp_path, monkeypatch, skip_movie=True):
    from cg_us.backends import base, chaperong
    from cg_us.config import Protocol
    from cg_us.manifest import Entry

    root = _fake_chaperong(tmp_path / "cg").parent.parent
    launcher = root / "CHAP_modules" / "run_CHAPERONg.sh"
    launcher.write_text(FAKE_CHAPERONG_STAGE0)
    launcher.chmod(0o755)

    wd = tmp_path / "systems" / "sys" / "rep1"
    wd.mkdir(parents=True)
    (wd.parent / "prep.json").write_text(json.dumps(
        {"center_nm": [1.0, 2.0, 3.0], "box_nm": [4.0, 5.0, 6.0]}))
    (wd / "sys.pdb").write_text("ATOM\n")

    proto = Protocol()
    proto.run.chaperong = str(launcher)
    proto.run.skip_movie = skip_movie
    proto.run.gpu = False
    proto.run.timeout_s = 30

    monkeypatch.setattr(chaperong, "_wire_restraints", lambda ctx: None)
    monkeypatch.setattr(chaperong, "_make_index", lambda ctx: None)

    entry = Entry(name="sys", pdb=Path("sys.pdb"), target="A", binder="B")
    return chaperong, base.RunContext(entry=entry, proto=proto, workdir=wd, replica=1)


def test_driver_stops_before_the_movie_and_resumes_at_stage_14(tmp_path, monkeypatch):
    pytest.importorskip("pexpect")
    chaperong, ctx = _stub_replica(tmp_path, monkeypatch, skip_movie=True)

    status = chaperong.run(ctx)

    assert [s["stage"] for s in status["sessions"]] == [0, 14]
    assert status["sessions"][0]["stopped_at"] == "stop before the SMD movie"
    assert not (ctx.workdir / "movie_marker.txt").exists()
    assert status["windows"] == 2
    assert (ctx.workdir / "index_answer.txt").read_text().strip() == "no"


def test_driver_can_still_render_the_movie(tmp_path, monkeypatch):
    pytest.importorskip("pexpect")
    chaperong, ctx = _stub_replica(tmp_path, monkeypatch, skip_movie=False)

    status = chaperong.run(ctx)

    assert [s["stage"] for s in status["sessions"]] == [0]
    assert (ctx.workdir / "movie_marker.txt").exists()


def test_entry_stage_resumes_after_a_finished_smd(tmp_path, monkeypatch):
    chaperong, ctx = _stub_replica(tmp_path, monkeypatch)
    assert chaperong.entry_stage(ctx) == 0

    (ctx.workdir / "pull.gro").write_text("x")
    assert chaperong.entry_stage(ctx) == 14

    (ctx.workdir / "tpr_files.dat").write_text("a.tpr\n")
    assert chaperong.entry_stage(ctx) is None


def test_stop_hook_refuses_a_truncated_smd(tmp_path, monkeypatch):
    chaperong, ctx = _stub_replica(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="steered MD did not finish"):
        chaperong._assert_smd_finished(ctx)


def test_pymol_shim_forces_headless_rendering(tmp_path, monkeypatch):
    chaperong, ctx = _stub_replica(tmp_path, monkeypatch, skip_movie=False)
    monkeypatch.setattr(chaperong.shutil, "which",
                        lambda n: "/usr/bin/pymol" if n == "pymol" else None)

    bindir = chaperong.write_pymol_shim(ctx)
    shim = bindir / "pymol"
    assert shim.name == "pymol" and os.access(shim, os.X_OK)
    assert 'exec "/usr/bin/pymol" -cq "$@"' in shim.read_text()


def test_pymol_shim_uses_xvfb_when_available(tmp_path, monkeypatch):
    chaperong, ctx = _stub_replica(tmp_path, monkeypatch, skip_movie=False)
    monkeypatch.setattr(chaperong.shutil, "which",
                        lambda n: {"pymol": "/usr/bin/pymol",
                                   "xvfb-run": "/usr/bin/xvfb-run"}.get(n))

    shim = chaperong.write_pymol_shim(ctx) / "pymol"
    assert '"/usr/bin/xvfb-run" -a "/usr/bin/pymol" -cq' in shim.read_text()


def test_no_pymol_shim_when_the_movie_is_skipped(tmp_path, monkeypatch):
    chaperong, ctx = _stub_replica(tmp_path, monkeypatch, skip_movie=True)
    assert chaperong.write_pymol_shim(ctx) is None


def test_offload_ladder_steps_down():
    from cg_us.backends.base import gpu_offload_ladder
    from cg_us.config import Protocol

    proto = Protocol()
    proto.run.gpu = True
    ladder = gpu_offload_ladder(proto)
    assert ladder[0][-2:] == ["-update", "gpu"]
    assert "-update" not in ladder[1]
    assert ladder[-1] == ["-nb", "gpu"]

    proto.run.gpu = False
    assert gpu_offload_ladder(proto) == [[]]


def test_shim_injects_cyclisation_and_the_mdrun_ladder(tmp_path):
    from cg_us.config import Protocol
    from cg_us.prep import write_gmx_shim

    proto = Protocol()
    proto.run.gpu = True
    shim = write_gmx_shim(tmp_path, proto).read_text()
    extra = [l for l in shim.splitlines() if l.startswith("PDB2GMX_EXTRA=")][0]
    assert "-sb" in extra and "-lb" in extra
    assert "-heavyh" not in extra        # HMR moved to the mdp option
    assert "-update gpu" in shim
    assert "MDRUN_LADDER=" in shim


def test_protocol_rejects_a_big_timestep_without_hmr():
    from cg_us.config import Protocol

    proto = Protocol()
    proto.validate()

    proto.prep.mass_repartition_factor = 1.0
    with pytest.raises(ValueError, match="hydrogen mass repartitioning"):
        proto.validate()


def test_expected_window_count_matches_the_selector():
    import numpy as np
    from cg_us.cli import _expected_windows
    from cg_us.config import Protocol

    proto = Protocol()
    u = proto.umbrella
    frames = np.arange(4000)
    dists = 0.5 + np.linspace(0, 3.0, 4000)
    picked = windows.select_windows(frames, dists, u.window_spacing, u.max_distance,
                                    dense_until=u.dense_until, dense_spacing=u.dense_spacing)
    assert abs(len(picked) - _expected_windows(proto)) <= 1
    assert len(picked) < 26


def test_benchmark_parses_performance(tmp_path, monkeypatch):
    from cg_us import bench

    (tmp_path / "pull.tpr").write_text("x")

    def fake_run(cmd, cwd, text, stdout, stderr):
        deffnm = Path(cmd[cmd.index("-deffnm") + 1])
        speed = 900.0 if "-update" in cmd else 400.0
        deffnm.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{deffnm}.log").write_text(f"Performance:      {speed}      0.027\n")
        return subprocess.CompletedProcess(cmd, 0, "done", "")

    import subprocess
    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    df = bench.benchmark(tmp_path, steps=100, concurrency=(1, 2), keep=True)

    assert df["ok"].all()
    assert df.loc[df["config"] == "gpu-resident", "ns_per_day"].iloc[0] == 900.0
    conc = df[df["concurrency"] == 2]
    assert conc["aggregate_ns_per_day"].iloc[0] == 1800.0

    rec = bench.recommend(df, windows=16, time_ns=10, replicas=3)
    assert rec["total_ns_per_system"] == 480
    assert rec["speedup_vs_nb_gpu"] == 4.5


def test_statistical_inefficiency_detects_correlation():
    from cg_us.convergence import statistical_inefficiency

    rng = np.random.default_rng(0)
    white = rng.normal(size=20000)
    assert statistical_inefficiency(white) < 3

    # AR(1) with phi=0.95 has g = (1+phi)/(1-phi) = 39
    correlated = np.zeros(40000)
    for i in range(1, len(correlated)):
        correlated[i] = 0.95 * correlated[i - 1] + rng.normal()
    g = statistical_inefficiency(correlated)
    assert 20 < g < 70


def test_window_diagnostics_and_verdict(tmp_path):
    from cg_us.convergence import verdict, window_diagnostics
    from cg_us.xvg import write_xvg

    rng = np.random.default_rng(1)
    t = np.arange(0, 5000, 1.0)

    good = tmp_path / "umbrella_win0_pullx.xvg"
    write_xvg(good, np.column_stack([t, 1.0 + rng.normal(0, 0.02, len(t))]))
    d = window_diagnostics(good, 0, discard_ps=1000)
    assert d["usable"] and d["n_eff"] > 500
    assert verdict(d, 50, 0.5)[0] is False

    drifting = tmp_path / "umbrella_win1_pullx.xvg"
    write_xvg(drifting, np.column_stack([t, 1.0 + np.linspace(0, 0.2, len(t))
                                         + rng.normal(0, 0.02, len(t))]))
    d2 = window_diagnostics(drifting, 1, discard_ps=1000)
    needs, why = verdict(d2, 50, 0.5)
    assert needs and "drift" in why


def test_budget_respects_the_ceiling():
    from cg_us.convergence import budget

    diag = [
        {"window": 0, "needs_more": False, "sampled_ns": 5.0},
        {"window": 1, "needs_more": True, "sampled_ns": 5.0, "reason": "x"},
        {"window": 2, "needs_more": True, "sampled_ns": 24.0, "reason": "y"},
        {"window": 3, "needs_more": True, "sampled_ns": 25.0, "reason": "z"},
    ]
    plan = {p["window"]: p for p in budget(diag, extend_ns=5.0, max_ns=25.0)}
    assert 0 not in plan
    assert plan[1]["extend_ns"] == 5.0
    assert plan[2]["extend_ns"] == 1.0
    assert plan[3]["extend_ns"] == 0.0 and plan[3]["capped"]


def test_survey_reads_the_window_list(tmp_path):
    from cg_us.convergence import survey
    from cg_us.xvg import write_xvg

    rng = np.random.default_rng(2)
    t = np.arange(0, 3000, 1.0)
    names = []
    for i in range(3):
        name = f"umbrella_win{i}_conf{i}_pullx.xvg"
        write_xvg(tmp_path / name, np.column_stack([t, 1.0 + 0.1 * i + rng.normal(0, 0.02, len(t))]))
        names.append(name)
    (tmp_path / "pullx_files.dat").write_text("\n".join(names) + "\n")

    diag = survey(tmp_path, discard_ps=500, min_neff=50, max_drift_sd=0.5)
    assert len(diag) == 3
    assert all(d["usable"] for d in diag)
    assert [d["window"] for d in diag] == [0, 1, 2]


def test_log_excerpt_surfaces_the_failure(tmp_path):
    from cg_us.backends import chaperong

    log = tmp_path / "chaperong_stage0.log"
    log.write_text(
        "\x1b[92m#=========#\x1b[m\n"
        "Solvating the system...\n"
        + "filler\n" * 40
        + "\x1b[31mFatal error:\x1b[m\n"
        "Atom OW in residue SOL 1 not found in rtp entry\n"
        + "trailing\n" * 5
    )
    text = chaperong.log_excerpt(log)
    assert "\x1b[" not in text
    assert "Fatal error:" in text
    assert "not found in rtp entry" in text
    assert "first sign of trouble" in text


def test_thread_flags_split_cores_between_workers():
    from cg_us.backends.base import thread_flags, threads_per_worker
    from cg_us.config import Protocol

    proto = Protocol()
    proto.run.cpu_cores = 32
    proto.run.ntomp = 0

    assert threads_per_worker(proto, 1) == 32
    assert threads_per_worker(proto, 4) == 8

    flags = thread_flags(proto, slot=2, workers=4)
    assert flags[:4] == ["-ntmpi", "1", "-ntomp", "8"]
    assert flags[4:] == ["-pin", "on", "-pinoffset", "16", "-pinstride", "1"]

    # an explicit ntomp is a ceiling, never an invitation to oversubscribe
    proto.run.ntomp = 16
    assert threads_per_worker(proto, 4) == 8
    assert threads_per_worker(proto, 1) == 16
    assert "-pin" not in thread_flags(proto, workers=1)


def test_shim_does_not_duplicate_flags_chaperong_already_passed(tmp_path):
    import subprocess
    from cg_us.config import Protocol
    from cg_us.prep import write_gmx_shim

    fake = tmp_path / "strict_gmx"
    fake.write_text(
        '#!/bin/bash\n'
        'seen=""\n'
        'for a in "$@"; do\n'
        '  case "$a" in -*)\n'
        '    case " $seen " in *" $a "*)\n'
        '      echo "Invalid command-line options: In command-line option $a Option specified multiple times" >&2\n'
        '      exit 1 ;; esac\n'
        '    seen="$seen $a" ;;\n'
        '  esac\n'
        'done\n'
        'echo "ran: $*"\n'
    )
    fake.chmod(0o755)

    proto = Protocol()
    proto.run.gmx = str(fake)
    proto.run.gpu = True
    shim = write_gmx_shim(tmp_path, proto)

    # this is exactly how CHAPERONg calls it when launched with -g
    out = subprocess.run([str(shim), "mdrun", "-ntmpi", "1", "-nb", "gpu", "-deffnm", "npt"],
                         text=True, capture_output=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("-nb") == 1
    assert "-update gpu" in out.stdout
    assert "-pme gpu" in out.stdout


def test_shim_still_supplies_everything_when_caller_passes_nothing(tmp_path):
    import subprocess
    from cg_us.config import Protocol
    from cg_us.prep import write_gmx_shim

    fake = tmp_path / "echo_gmx"
    fake.write_text('#!/bin/bash\necho "ran: $*"\n')
    fake.chmod(0o755)

    proto = Protocol()
    proto.run.gmx = str(fake)
    proto.run.gpu = True
    shim = write_gmx_shim(tmp_path, proto)

    out = subprocess.run([str(shim), "mdrun", "-deffnm", "umbrella_win0_conf0"],
                         text=True, capture_output=True)
    assert "-nb gpu" in out.stdout and "-update gpu" in out.stdout


def test_parafile_always_gives_chaperong_a_thread_count(tmp_path):
    from cg_us.config import Protocol
    from cg_us.mdp import write_parafile

    proto = Protocol()
    proto.run.ntomp = 0
    proto.run.ntmpi = 0
    proto.run.cpu_cores = 56
    write_parafile(tmp_path, proto)

    text = (tmp_path / "paraFile.par").read_text()
    assert "ntmpi             =      1" in text
    assert "ntomp             =      56" in text


def test_minimisation_gets_no_pme_or_update_offload(tmp_path):
    import subprocess
    from cg_us.backends.base import gpu_offload_ladder
    from cg_us.config import Protocol
    from cg_us.prep import write_gmx_shim

    proto = Protocol()
    proto.run.gpu = True
    assert gpu_offload_ladder(proto, dynamical=False) == [["-nb", "gpu"], []]

    fake = tmp_path / "gmx"
    fake.write_text('#!/bin/bash\necho "ran: $*"\n')
    fake.chmod(0o755)
    proto.run.gmx = str(fake)
    shim = write_gmx_shim(tmp_path, proto)

    em = subprocess.run([str(shim), "mdrun", "-nb", "gpu", "-deffnm", "em"],
                        text=True, capture_output=True).stdout
    assert "-pme" not in em and "-update" not in em

    md = subprocess.run([str(shim), "mdrun", "-nb", "gpu", "-deffnm", "nvt"],
                        text=True, capture_output=True).stdout
    assert "-pme gpu" in md and "-update gpu" in md


def test_pme_integrator_complaint_is_treated_as_a_rejection():
    from cg_us.backends.base import _offload_rejected

    real = ("Inconsistency in user input:\n"
            "Cannot compute PME interactions on a GPU, because:\n"
            "  PME GPU does not support:\n    Non-dynamical integrator (use md, sd, etc).")
    assert _offload_rejected(real)
    assert not _offload_rejected("Steepest Descents converged to Fmax < 500 in 842 steps")


def test_gmx_log_method_survives_on_the_class(tmp_path):
    from cg_us.backends.base import Gmx

    log = tmp_path / "gmx.log"
    g = Gmx("echo", tmp_path, log)
    g.run(["hello"])
    assert "hello" in log.read_text()


def test_box_accounts_for_the_pull_minimum_image_check():
    """Regression: actrIIb died at 90% of the SMD because d0 was ignored."""
    from cg_us.config import Protocol
    from cg_us.prep import PULL_BOX_FRACTION, max_pull_distance, pull_box_length

    proto = Protocol()
    extent_z, d0 = 3.591, 1.71          # measured from 5NGV after orientation
    pull = proto.umbrella.max_distance

    old_box = extent_z + pull + 2 * proto.prep.box_edge_z
    assert PULL_BOX_FRACTION * old_box * 0.98 < d0 + pull      # what actually happened

    box_z, driver = pull_box_length(extent_z, d0, proto)
    assert driver == "pull minimum-image check"
    assert box_z > old_box
    # survives a 2% NPT compression with room to spare
    assert PULL_BOX_FRACTION * box_z * 0.98 > d0 + pull
    assert max_pull_distance(box_z, d0, proto) == pytest.approx(pull, abs=1e-6)


def test_geometry_still_wins_for_elongated_complexes():
    from cg_us.config import Protocol
    from cg_us.prep import max_pull_distance, pull_box_length

    proto = Protocol()
    # long complex, binder sitting close to the target COM: geometry dominates
    box_z, driver = pull_box_length(extent_z := 8.0, 1.0, proto)
    assert driver == "geometry"
    assert box_z == pytest.approx(extent_z + 2.0 + 2.0)
    assert max_pull_distance(box_z, 1.0, proto) > proto.umbrella.max_distance


def test_every_bench6_system_can_complete_its_pull():
    from cg_us.config import Protocol
    from cg_us.prep import max_pull_distance, pull_box_length

    proto = Protocol()
    measured = {"actrIIb": (3.591, 1.71), "keap1": (5.717, 2.32), "mdm2_p53": (3.201, 1.11),
                "sfti": (4.455, 1.55), "mcoti": (4.656, 1.87), "pmi": (3.402, 1.22)}
    for name, (extent_z, d0) in measured.items():
        box_z, _ = pull_box_length(extent_z, d0, proto)
        assert max_pull_distance(box_z, d0, proto) >= proto.umbrella.max_distance - 1e-6, name


def test_headroom_covers_drift_and_window_fluctuation():
    """Regression: window 11 hit 3.886 nm where prep budgeted 3.710."""
    import math
    from cg_us.config import Protocol
    from cg_us.prep import PULL_BOX_FRACTION, max_pull_distance, pull_box_length

    proto = Protocol()
    extent_z, d0_crystal, observed = 3.591, 1.71, 3.886

    box_z, driver = pull_box_length(extent_z, d0_crystal, proto)
    assert driver == "pull minimum-image check"
    # survives the observed excursion even after the 1.2% NPT compression seen in the log
    assert PULL_BOX_FRACTION * box_z * 0.988 > observed

    kT = 8.314462618e-3 * proto.temperature
    sigma = math.sqrt(kT / proto.umbrella.k)
    assert proto.prep.pull_headroom > 4 * sigma      # room for the outermost window to rattle
    assert max_pull_distance(box_z, d0_crystal, proto) == pytest.approx(
        proto.umbrella.max_distance, abs=1e-6)


def test_pull_range_check_shortens_an_overlong_pull(tmp_path, monkeypatch):
    from cg_us.backends import base
    from cg_us.config import Protocol
    from cg_us.manifest import Entry
    from cg_us.mdp import write_all

    proto = Protocol()
    proto.run.gmx = "true"
    entry = Entry(name="x", pdb=Path("x.pdb"), target="A", binder="B")
    ctx = base.RunContext(entry=entry, proto=proto, workdir=tmp_path, replica=1)
    write_all(tmp_path, proto, restrain_binder=False, seed=1)
    before = [l for l in (tmp_path / "md_pull.mdp").read_text().splitlines()
              if l.startswith("nsteps")][0]

    # a box that only leaves ~1.2 nm of room once the drifted COM distance is in
    (tmp_path / "npt.gro").write_text("x\n2\n" + "line\n" * 2 + "  6.00000  6.00000  7.00000\n")
    monkeypatch.setattr(base.Gmx, "run",
                        lambda self, args, stdin=None, check=True: __import__("subprocess")
                        .CompletedProcess(args, 0, "", ""))
    monkeypatch.setattr(base, "read_xvg", None, raising=False)
    monkeypatch.setattr("cg_us.xvg.read_xvg", lambda p: (np.array([[0.0, 2.20]]), {}))

    info = base.verify_pull_range(ctx)
    assert info["com_distance_after_npt_nm"] == 2.2
    assert info["max_distance_effective"] < proto.umbrella.max_distance
    assert base.effective_max_distance(ctx) == info["max_distance_effective"]

    after = [l for l in (tmp_path / "md_pull.mdp").read_text().splitlines()
             if l.startswith("nsteps")][0]
    assert after != before
    assert int(after.split("=")[1]) == int(info["max_distance_effective"] / 0.001 / 0.004)


def test_hmr_uses_the_guarded_mdp_option_not_heavyh(tmp_path):
    """Regression: -heavyh left methyl carbons at 2.9 amu and NVT blew up at 19.6 ps."""
    from cg_us.config import Protocol
    from cg_us.mdp import write_all
    from cg_us.prep import write_gmx_shim

    proto = Protocol()
    assert proto.prep.heavy_hydrogens is False
    assert proto.prep.mass_repartition_factor == 3.0
    proto.validate()

    write_all(tmp_path, proto, restrain_binder=False, seed=1)
    for name in ("nvt.mdp", "npt.mdp", "md_pull.mdp", "npt_umbrella.mdp", "md_umbrella.mdp"):
        text = (tmp_path / name).read_text()
        assert "mass-repartition-factor  = 3.0" in text, name
        assert "dt                       = 0.004" in text, name

    proto.run.gmx = "gmx"
    extra = [l for l in write_gmx_shim(tmp_path, proto).read_text().splitlines()
             if l.startswith("PDB2GMX_EXTRA=")][0]
    assert "-heavyh" not in extra


def test_protocol_rejects_the_two_hmr_schemes_together():
    from cg_us.config import Protocol

    proto = Protocol()
    proto.prep.heavy_hydrogens = True
    with pytest.raises(ValueError, match="lighter than their own hydrogens"):
        proto.validate()

    proto.prep.heavy_hydrogens = False
    proto.prep.mass_repartition_factor = 1.0
    with pytest.raises(ValueError, match="mass_repartition_factor"):
        proto.validate()

    proto.smd.dt = proto.umbrella.dt = 0.002
    proto.validate()


def test_seeds_are_stable_across_processes():
    """hash() on str is salted per interpreter, so prep gave new velocities each time."""
    import subprocess
    import sys

    code = ("import zlib;print(20181 + 1000*1 + zlib.crc32(b'mdm2_p53') % 997)")
    seeds = {subprocess.run([sys.executable, "-c", code], text=True,
                            capture_output=True).stdout.strip() for _ in range(3)}
    assert len(seeds) == 1


def test_multi_model_pdb_keeps_first_model(tmp_path):
    body = _pdb(tmp_path).read_text().replace("END\n", "")
    p = tmp_path / "ensemble.pdb"
    p.write_text("MODEL        1\n" + body + "ENDMDL\nMODEL        2\n" + body + "ENDMDL\nEND\n")
    single = structure.read_pdb(_pdb(tmp_path))
    assert len(structure.read_pdb(p)) == len(single)


CYCLIC_ITP = """
[ moleculetype ]
Protein_chain_B     3

[ atoms ]
     1   NH1   1   SER   N    1  -0.47
     2     H   1   SER   HN   2   0.31
     3   CT1   1   SER   CA   3   0.07
     4    CC   2   GLY   C    4   0.51
     5    NH1  2   GLY   N    5  -0.47

[ bonds ]
    1     3     1
    4     5     1
    1     4     1
"""


def test_is_cyclic_detects_the_ring_bond(tmp_path):
    p = tmp_path / "chain.itp"
    p.write_text(CYCLIC_ITP)
    mt = topology.moltype(p, "Protein_chain_B")
    assert topology.is_cyclic(mt)
    p.write_text(CYCLIC_ITP.replace("    1     4     1\n", ""))
    assert not topology.is_cyclic(topology.moltype(p, "Protein_chain_B"))


def test_cyclic_binder_is_written_first(tmp_path):
    from cg_us.prep import _chain_order
    from cg_us.manifest import Entry

    entry = Entry(pdb=tmp_path / "x.pdb", name="x", target="A", binder="B")

    def report(cyclic):
        return structure.ChainReport("_", 1, 1, 1, 1, 0.13, cyclic, [], [])

    assert _chain_order(entry, {"A": report(False), "B": report(True)}) == ["B", "A"]
    assert _chain_order(entry, {"A": report(False), "B": report(False)}) == ["A", "B"]


def _fake_ctx(tmp_path):
    from cg_us.backends.base import RunContext
    from cg_us.config import Protocol
    from cg_us.manifest import Entry
    entry = Entry(pdb=tmp_path / "x.pdb", name="x", target="A", binder="B")
    return RunContext(entry=entry, proto=Protocol(), workdir=tmp_path, replica=1)


def test_pbcatom_is_the_spatially_central_atom(tmp_path):
    from cg_us.backends.base import pull_pbcatoms, write_pull_pbcatoms

    xs = [0.0, 1.0, 2.0, 9.0, 9.5, 10.0]
    lines = ["fake", f"{len(xs)}"]
    for i, x in enumerate(xs, start=1):
        lines.append(f"{1:5d}ALA  {'CA':>5s}{i:5d}{x:8.3f}{0.0:8.3f}{0.0:8.3f}")
    lines.append("  20.00000  20.00000  20.00000")
    (tmp_path / "solv_ions.gro").write_text("\n".join(lines) + "\n")
    (tmp_path / "index.ndx").write_text("[ Target ]\n1 2 3\n[ Binder ]\n4 5 6\n")

    picked = pull_pbcatoms(_fake_ctx(tmp_path))
    assert picked == {"Target": 2, "Binder": 5}

    (tmp_path / "md_umbrella.mdp").write_text(
        "pull_group1_pbcatom = 999\npull_pbc_ref_prev_step_com = yes\n")
    write_pull_pbcatoms(_fake_ctx(tmp_path), picked)
    text = (tmp_path / "md_umbrella.mdp").read_text()
    assert "pull_group1_pbcatom = 2" in text
    assert "pull_group2_pbcatom = 5" in text
    assert "999" not in text


def test_minimisation_guard_rejects_infinite_force(tmp_path):
    from cg_us.backends.base import check_minimisation, GmxError
    (tmp_path / "em.log").write_text("Maximum force     =            inf on atom 1459\n")
    with pytest.raises(GmxError, match="broken topology"):
        check_minimisation(_fake_ctx(tmp_path))
    (tmp_path / "em.log").write_text("Maximum force     =  4.3298151e+02 on atom 2205\n")
    assert check_minimisation(_fake_ctx(tmp_path)) == pytest.approx(432.98, abs=0.01)


def test_window_failure_is_isolated_and_retried(monkeypatch, tmp_path):
    from cg_us.backends import direct
    from cg_us.backends.base import GmxError

    calls = []

    def fake(ctx, window, frame, slot=0, workers=1, retry=0, dt_scale=1.0):
        calls.append((window, retry))
        if window == 3 and retry == 0:
            raise GmxError("Step 660100: The total potential energy is nan")
        return {"window": window, "frame": frame}

    monkeypatch.setattr(direct, "_run_window", fake)
    ctx = _fake_ctx(tmp_path)
    ok = direct._guarded_window(ctx, {"window": 3, "frame": 918}, 0, 1)
    assert "error" not in ok
    assert calls == [(3, 0), (3, 1)]

    ctx.proto.run.window_retries = 0
    bad = direct._guarded_window(ctx, {"window": 3, "frame": 918}, 0, 1)
    assert "nan" in bad["error"]


def test_fill_gaps_adds_one_window_per_broken_pair(monkeypatch, tmp_path):
    from cg_us.backends import direct

    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "replica_result.json").write_text(json.dumps({
        "detail_overlap": {
            "overlaps": [0.15, 0.0008, 0.12],
            "window_centers_nm": [0.5, 0.9, 1.3, 1.7],
        }
    }))
    frames = np.arange(50)
    dists = 0.5 + np.linspace(0, 1.5, 50)
    windows.write_distance_summary(tmp_path / "distances_summary.txt", frames, dists)
    existing = [
        {"window": 0, "frame": 0, "target_distance": 0.5, "distance": 0.5},
        {"window": 1, "frame": 13, "target_distance": 0.9, "distance": 0.9},
        {"window": 2, "frame": 26, "target_distance": 1.3, "distance": 1.3},
        {"window": 3, "frame": 40, "target_distance": 1.7, "distance": 1.7},
    ]
    (tmp_path / "windows.json").write_text(json.dumps({"windows": existing}))
    old_records = [{"window": w["window"], "frame": w["frame"],
                    "tpr": f"umbrella_win{w['window']}.tpr",
                    "pullf": f"umbrella_win{w['window']}_pullf.xvg",
                    "pullx": f"umbrella_win{w['window']}_pullx.xvg"} for w in existing]
    (tmp_path / "window_records.json").write_text(json.dumps(old_records))

    calls = []

    def fake_run_window(ctx, window, frame, slot=0, workers=1, retry=0, dt_scale=1.0):
        calls.append((window, frame))
        return {"window": window, "frame": frame,
                "tpr": f"umbrella_win{window}.tpr",
                "pullf": f"umbrella_win{window}_pullf.xvg",
                "pullx": f"umbrella_win{window}_pullx.xvg"}

    monkeypatch.setattr(direct, "_run_window", fake_run_window)
    ctx = _fake_ctx(tmp_path)
    added = direct.fill_gaps(ctx)

    assert len(added) == 1
    assert added[0]["window"] == 4  # next free index after 0-3
    assert calls == [(4, added[0]["frame"])]
    assert added[0]["gap_before_nm"] == 0.9
    assert added[0]["gap_after_nm"] == 1.3
    # the new window's target sits at the gap midpoint
    new_entry = json.loads((tmp_path / "windows.json").read_text())["windows"][-1]
    assert new_entry["target_distance"] == pytest.approx(1.1)
    # file lists and window_records.json now include the new window too
    records = json.loads((tmp_path / "window_records.json").read_text())
    assert len(records) == 5
    assert (tmp_path / "tpr_files.dat").read_text().count("\n") == 5
    assert f"umbrella_win4.tpr" in (tmp_path / "tpr_files.dat").read_text()

    # a second call, unchanged analysis, is idempotent about which frame it
    # would reuse - the previously-picked frame must not be picked again
    added2 = direct.fill_gaps(ctx)
    if added2:
        assert added2[0]["frame"] != added[0]["frame"]


def test_fill_gaps_requires_analysis_first(tmp_path):
    from cg_us.backends import direct
    from cg_us.backends.base import GmxError
    ctx = _fake_ctx(tmp_path)
    with pytest.raises(GmxError, match="cg-us analyze"):
        direct.fill_gaps(ctx)


def test_fill_gaps_returns_nothing_below_threshold(monkeypatch, tmp_path):
    from cg_us.backends import direct
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "replica_result.json").write_text(json.dumps({
        "detail_overlap": {"overlaps": [0.15, 0.12], "window_centers_nm": [0.5, 0.9, 1.3]}
    }))
    monkeypatch.setattr(direct, "_run_window", lambda *a, **k: pytest.fail("should not run"))
    ctx = _fake_ctx(tmp_path)
    assert direct.fill_gaps(ctx) == []


def test_retry_mdp_halves_the_step_and_reseeds(tmp_path):
    from cg_us.backends import direct
    ctx = _fake_ctx(tmp_path)
    (tmp_path / "md_umbrella.mdp").write_text(
        "dt                       = 0.004\nnsteps                   = 1250000\n"
        "continuation             = yes\nconstraints              = h-bonds\n")
    name = direct._retry_inputs(ctx, "win3_conf918", 0.5, 4242)
    text = (tmp_path / name).read_text()
    assert "dt                       = 0.002" in text
    assert f"nsteps                   = {int(ctx.proto.umbrella.time_ns * 1000 / 0.002)}" in text
    assert "continuation             = no" in text
    assert "gen_vel                  = yes" in text
    assert "constraints              = h-bonds" in text


def test_four_femtoseconds_requires_all_bonds():
    from cg_us.config import Protocol

    proto = Protocol()
    assert proto.prep.constraints == "all-bonds"
    proto.validate()

    proto.prep.constraints = "h-bonds"
    with pytest.raises(ValueError, match="heavy-atom bonds unconstrained"):
        proto.validate()

    proto.umbrella.dt = proto.smd.dt = 0.003
    proto.validate()


ETHERS_NTDB = """[ None ]

[ HYD1 ]
[ replace ]
    HN  H  1.008  0.09

[ MET1 ]
; Append terminal methyl group adjacent to CH2
[ delete ]
 H1C
[ add ]
1   5   CE    C1     H1A    H1B    O1
  CC33A     12.011000  -0.2700  -1
"""


def test_prune_terminus_db_drops_the_ether_met_entry(tmp_path):
    from cg_us.prep import prune_terminus_db

    ff = tmp_path / "charmm36.ff"
    ff.mkdir()
    (ff / "ethers.n.tdb").write_text(ETHERS_NTDB)
    (ff / "aminoacids.n.tdb").write_text("[ None ]\n\n[ NH3+ ]\n[ add ]\n1 1 H N CA\n")

    removed = prune_terminus_db(ff)
    assert removed == ["ethers.n.tdb:[MET1]"]

    text = (ff / "ethers.n.tdb").read_text()
    assert "MET1" not in text
    assert "[ HYD1 ]" in text and "[ None ]" in text
    assert "C1" not in text
    # the amino-acid database is never touched, even though its names start with codes
    assert "[ NH3+ ]" in (ff / "aminoacids.n.tdb").read_text()


def test_forcefield_dirs_finds_the_ff_inside_a_parent(tmp_path):
    from cg_us.prep import forcefield_dirs

    parent = tmp_path / "ff"
    (parent / "charmm36-jul2022.ff").mkdir(parents=True)
    assert [p.name for p in forcefield_dirs(parent)] == ["charmm36-jul2022.ff"]
    assert forcefield_dirs(parent / "charmm36-jul2022.ff")[0].name == "charmm36-jul2022.ff"


def test_error_excerpt_finds_the_fatal_block_not_the_tail():
    from cg_us.backends.base import error_excerpt

    out = ("noise\n" * 20 + "Fatal error:\natom C1 not found in buiding block 1MET\n"
           + "buffered stdout\n" * 80)
    excerpt = error_excerpt(out)
    assert "atom C1 not found" in excerpt
    assert "more lines in gmx.log" in excerpt


def test_pinned_mdp_replaces_the_floating_reference(tmp_path):
    from cg_us.backends import direct

    ctx = _fake_ctx(tmp_path)
    (tmp_path / "md_umbrella.mdp").write_text(
        "dt                       = 0.004\n"
        "pull_coord1_start        = yes\n"
        "pull_coord1_init         = 0\n"
        "pull_coord1_k            = 1000.0\n")
    name = direct._pinned_mdp(ctx, "md_umbrella.mdp", "win7_conf123", 2.4567)
    text = (tmp_path / name).read_text()
    assert name == "md_umbrella_win7_conf123.mdp"
    assert "pull_coord1_start        = no" in text
    assert "pull_coord1_init         = 2.4567" in text
    assert "pull_coord1_start        = yes" not in text
    assert text.count("pull_coord1_init") == 1
    assert "pull_coord1_k            = 1000.0" in text


def test_window_targets_uses_the_planned_grid(tmp_path):
    import json as _json
    from cg_us.backends import direct

    ctx = _fake_ctx(tmp_path)
    assert direct.window_targets(ctx) == {}
    (tmp_path / "windows.json").write_text(_json.dumps({"windows": [
        {"window": 0, "frame": 0, "target_distance": 1.2, "distance": 1.207},
        {"window": 1, "frame": 40, "target_distance": 1.3, "distance": 1.298},
        {"window": 2, "frame": 90, "distance": 1.401},
    ]}))
    assert direct.window_targets(ctx) == {0: 1.2, 1: 1.3, 2: 1.401}


def test_distances_prefer_the_pull_coordinate(tmp_path, monkeypatch):
    from cg_us.backends import direct

    ctx = _fake_ctx(tmp_path)
    (tmp_path / "coordinates_SMD").mkdir()
    for i in range(3):
        (tmp_path / "coordinates_SMD" / f"coordinate{i}.gro").touch()
    (tmp_path / "pullx.xvg").write_text(
        "# comment\n@ title\n0.0 1.500\n2.0 1.602\n4.0 1.703\n")

    called = []
    monkeypatch.setattr(type(ctx.gmx), "run",
                        lambda self, *a, **k: called.append(a) or None)
    direct._distances(ctx)
    assert not called, "gmx distance should not be needed when pullx.xvg is there"
    body = (tmp_path / "distances_summary.txt").read_text()
    assert "1.5" in body and "1.703" in body


def _pmf(xi, g, err=None):
    from cg_us.wham import PMF
    return PMF(xi=np.asarray(xi), energy=np.asarray(g),
               error=None if err is None else np.asarray(err))


def test_connected_range_stops_at_the_first_thin_pair():
    from cg_us.wham import connected_range

    ov = {"overlaps": [0.31, 0.28, 0.002, 0.30], "window_centers_nm": [1.0, 1.1, 1.2, 1.3, 1.4]}
    assert connected_range(ov, 0.03) == 1.2      # the last window still tied to its neighbour
    assert connected_range(ov, 0.001) is None    # nothing below the threshold
    assert connected_range({"overlaps": [], "window_centers_nm": []}, 0.03) is None


def _well(depth=-8.0, step_at=None, step=0.0):
    """A well that has clearly flattened by 1.9 nm, optionally with a WHAM step in the tail."""
    xi = np.linspace(1.0, 5.0, 401)
    g = depth * np.exp(-((xi - 1.2) / 0.2) ** 2)
    g = g - g[-20:].mean()
    if step_at is not None:
        g = g + np.where(xi > step_at, step, 0.0)
    return xi, g


def test_truncation_removes_the_offset_a_gap_injected():
    from cg_us.wham import binding_free_energy

    xi, g = _well(step_at=3.0, step=6.0)          # 6 kcal/mol jump past the disconnected pair
    full = binding_free_energy(_pmf(xi, g), plateau_width=0.4)
    cut = binding_free_energy(_pmf(xi, g), plateau_width=0.4, xi_max=2.95)

    assert full["dG"] == pytest.approx(-14.0, abs=0.2)   # contaminated by the step
    assert cut["dG"] == pytest.approx(-8.0, abs=0.2)     # the connected stretch alone
    assert cut["truncated_at_gap"] and cut["xi_connected_nm"] == 2.95
    assert cut["xi_range_full_nm"] == [1.0, 5.0]
    assert full["truncated_at_gap"] is False


def test_no_number_when_the_cut_lands_on_the_rise():
    from cg_us.wham import binding_free_energy

    xi, g = _well()
    cut = binding_free_energy(_pmf(xi, g), plateau_width=0.4, xi_max=2.0)
    assert cut["dG"] is None
    assert "still rising" in cut["no_estimate_reason"]
    assert cut["truncated_at_gap"]

    tiny = binding_free_energy(_pmf(xi, g), plateau_width=0.4, xi_max=1.05)
    assert tiny["dG"] is None and tiny["no_estimate_reason"]



def _rep(replica, dG, reason=None):
    return {"system": "x", "replica": replica, "dG": dG, "dG_bootstrap_error": 0.5,
            "overlap_min": 0.0, "n_windows": 22, "no_estimate_reason": reason,
            "truncated_at_gap": True, "xi_connected_nm": 2.4, "convergence": {},
            "plateau_roughness": 0.2, "xi_range_nm": [1.0, 2.4], "sampled_ns": 99.0}


def test_aggregate_uses_only_the_replicas_that_produced_a_number(tmp_path):
    from cg_us.analysis import aggregate_system
    from cg_us.config import Protocol
    from cg_us.manifest import Entry

    entry = Entry(pdb=tmp_path / "x.pdb", name="x", target="A", binder="B")
    proto = Protocol()

    s = aggregate_system([_rep(1, -8.0), _rep(2, -8.4),
                          _rep(3, None, "still rising at the end")], entry, proto)
    assert s["n_replicas"] == 2 and s["n_replicas_run"] == 3
    assert s["dG_mean"] == pytest.approx(-8.2, abs=0.01)
    assert s["replicas_without_estimate"] == [{"replica": 3, "reason": "still rising at the end"}]

    none = aggregate_system([_rep(i, None, "no plateau") for i in (1, 2, 3)], entry, proto)
    assert none["dG_mean"] is None and none["n_replicas"] == 0
    assert all("no plateau" in f for f in none["quality_flags"])


def test_systems_without_a_number_stay_out_of_the_regression(tmp_path):
    from cg_us import experiment
    from cg_us.config import Protocol

    proto = Protocol()
    summaries = [
        {"system": "a", "role": "benchmark", "dG_mean": -9.0, "dG_sd": 0.4, "dg_exp": -8.5,
         "error_vs_exp": -0.5},
        {"system": "b", "role": "benchmark", "dG_mean": -11.0, "dG_sd": 0.4, "dg_exp": -11.6,
         "error_vs_exp": 0.6},
        {"system": "c", "role": "benchmark", "dG_mean": None, "dG_sd": None, "dg_exp": -10.0,
         "error_vs_exp": None},
    ]
    out = experiment.compare(summaries, proto)
    assert out["n_in_regression"] == 2
    assert out["n_without_estimate"] == 1
    assert out["systems_without_estimate"] == ["c"]


def _write_window(tmp_path, tag, xi_ref, grad_kj, k=1000.0, n=3000, seed=0, sd=0.051):
    """A window sitting where dG/dxi equals `grad_kj` (kJ/mol/nm) at its mean.

    The umbrella balances the underlying gradient, so <xi> - xi_ref = -grad/k.
    """
    from cg_us.xvg import write_xvg

    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    xi = xi_ref - grad_kj / k + rng.normal(0, sd, n)
    f = -k * (xi - xi_ref)
    write_xvg(tmp_path / f"umbrella_{tag}_pullx.xvg", np.column_stack([t, xi]))
    write_xvg(tmp_path / f"umbrella_{tag}_pullf.xvg", np.column_stack([t, f]))
    (tmp_path / f"md_umbrella_{tag}.mdp").write_text(
        f"pull_coord1_start        = no\npull_coord1_init         = {xi_ref}\n")


def test_window_force_recovers_the_gradient(tmp_path):
    from cg_us.integration import window_force

    _write_window(tmp_path, "win3_conf99", xi_ref=1.5, grad_kj=40.0, seed=3)
    w = window_force(tmp_path / "umbrella_win3_conf99_pullf.xvg",
                     tmp_path / "umbrella_win3_conf99_pullx.xvg", window=3, xi_ref=1.5)
    assert w.grad == pytest.approx(40.0 / 4.184, abs=0.3)    # kcal/mol/nm
    assert w.xi_mean == pytest.approx(1.46, abs=0.01)
    assert w.k_measured == pytest.approx(1000.0, rel=1e-6)   # recovered from the data
    assert abs(w.ref_residual) < 1e-4                        # and it matches the mdp
    assert w.n_eff > 100 and not w.bimodal


def test_a_hole_in_the_ladder_costs_shape_not_an_offset(tmp_path):
    """The same analytic well integrated with and without one missing node."""
    from cg_us.integration import profile_from_windows

    depth, centre, width = -8.0, 1.2, 0.2          # kcal/mol Gaussian well
    def dG(x):
        return -2 * depth * (x - centre) / width ** 2 * np.exp(-((x - centre) / width) ** 2)

    def build(nodes, sub):
        d = tmp_path / sub
        d.mkdir()
        for i, x in enumerate(nodes):
            # the umbrella must be offset so the window actually settles on x
            grad_kj = dG(x) * 4.184
            _write_window(d, f"win{i}_conf{i * 7}", xi_ref=x + grad_kj / 1000.0,
                          grad_kj=grad_kj, seed=i)
        (d / "pullx_files.dat").write_text(
            "\n".join(f"umbrella_win{i}_conf{i * 7}_pullx.xvg" for i in range(len(nodes))) + "\n")
        return profile_from_windows(d, discard_ps=0.0, k=1000.0)

    full_nodes = [round(v, 3) for v in np.arange(1.0, 2.6, 0.1)]
    holed_nodes = [x for i, x in enumerate(full_nodes) if i != 8]   # 0.2 nm hole at 1.8 nm

    full = build(full_nodes, "full")
    holed = build(holed_nodes, "holed")
    assert len(full.windows) == 16 and len(holed.windows) == 15

    grid = np.linspace(1.05, 2.5, 300)
    a = np.interp(grid, full.xi, full.energy - full.energy.min())
    b = np.interp(grid, holed.xi, holed.energy - holed.energy.min())
    assert np.abs(a - b).max() < 0.3            # the hole barely moves the profile
    assert abs(a[-1] - b[-1]) < 0.2             # and leaves the plateau - i.e. dG - intact

    exact = depth * np.exp(-((grid - centre) / width) ** 2)
    assert np.abs(b - (exact - exact.min())).max() < 1.0   # trapezoid on a steep well


def test_agreement_measures_shape_not_offset():
    from cg_us.integration import agreement, ForceProfile

    xi = np.linspace(1.0, 3.0, 200)
    g = -8.0 * np.exp(-((xi - 1.2) / 0.2) ** 2)
    ui = ForceProfile(xi=xi, energy=g + 17.0, error=np.zeros_like(xi), windows=[])
    same = agreement(xi, g, ui)
    assert same["rms"] == pytest.approx(0.0, abs=1e-6)   # a constant offset is not a disagreement

    ui_bad = ForceProfile(xi=xi, energy=g * 1.5, error=np.zeros_like(xi), windows=[])
    assert agreement(xi, g, ui_bad)["rms"] > 1.0


def test_estimator_choice_needs_wham_agreement_first():
    from cg_us.analysis import _pick_estimator
    from cg_us.config import Protocol

    proto = Protocol()
    wham = {"dG": -14.0, "no_estimate_reason": None}
    ui = {"dG": -8.0, "dG_bootstrap_error": 0.4}

    # clean ladder: nothing to rescue, WHAM stands
    dg, src = _pick_estimator(proto, wham, ui, {"rms": 0.3}, None)
    assert src == "wham" and dg["dG"] == -14.0

    # a break, and the two agree where WHAM is sound -> the force profile takes over
    dg, src = _pick_estimator(proto, wham, ui, {"rms": 0.3}, 2.4)
    assert src == "umbrella_integration" and dg["dG"] == -8.0

    # a break, but they disagree even on the connected stretch -> no rescue, and say why
    dg, src = _pick_estimator(proto, wham, ui, {"rms": 2.7}, 2.4)
    assert src == "wham" and "disagrees with WHAM" in dg["no_estimate_reason"]

    proto.analysis.estimator = "umbrella_integration"
    assert _pick_estimator(proto, wham, ui, {"rms": 9.9}, None)[1] == "umbrella_integration"


def test_convergence_survives_halves_without_a_plateau(tmp_path, monkeypatch):
    """A short block that never reaches the plateau must not take the analysis down."""
    from cg_us import wham as W
    from cg_us.config import Protocol

    xi = np.linspace(1.0, 5.0, 401)
    deep = -8.0 * np.exp(-((xi - 1.2) / 0.2) ** 2)
    deep = deep - deep[-20:].mean()

    calls = {"n": 0}

    def fake_run(workdir, proto, prefix="wham", begin_ps=None, end_ps=None, bootstraps=None):
        calls["n"] += 1
        return {"prefix": prefix}

    def fake_load(paths):
        # the halves get a profile that is still climbing, the blocks a complete one
        if paths["prefix"].endswith("_half"):
            keep = xi <= 2.0
            return W.PMF(xi[keep], deep[keep])
        return W.PMF(xi, deep)

    monkeypatch.setattr(W, "run_wham", fake_run)
    monkeypatch.setattr(W, "load_pmf", fake_load)

    out = W.convergence(tmp_path, Protocol(), n_blocks=2)
    assert out["half_split_drift"] is None
    assert all(h["dG"] is None for h in out["halves"])
    assert all("rising" in h["reason"] for h in out["halves"])
    assert out["blocks_scored"] == 2


def test_truncation_slices_the_bootstrap_errors_too():
    """Regression: the error array kept its full length and the plateau mask overran it."""
    from cg_us.wham import binding_free_energy

    xi, g = _well(step_at=3.0, step=6.0)
    err = np.linspace(0.1, 0.9, xi.size)          # distinguishable per point
    cut = binding_free_energy(_pmf(xi, g, err), plateau_width=0.4, xi_max=2.95)

    assert cut["dG"] == pytest.approx(-8.0, abs=0.2)
    # the reported error must come from the kept part, not from the discarded tail
    kept = err[xi <= 2.95]
    assert cut["dG_bootstrap_error"] <= float(np.sqrt(2) * kept.max())
    assert cut["dG_bootstrap_error"] < binding_free_energy(
        _pmf(xi, g, err), plateau_width=0.4)["dG_bootstrap_error"]
