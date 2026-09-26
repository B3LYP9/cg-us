"""Contact geometry on hand-built configurations, and the trajectory pipeline on a tiny one."""

import json

import numpy as np
import pytest

from cg_us import contacts as C

BOX = np.array([40.0, 40.0, 40.0])


def make(atoms):
    """atoms: (resindex, resname, atom, side, (x,y,z)) -> (System, positions)."""
    pos = np.array([a[4] for a in atoms], float)
    sysm = C.build_system([a[2] for a in atoms], [a[1] for a in atoms], [a[0] for a in atoms],
                          [a[3] for a in atoms], pos)
    return sysm, pos


def found(atoms, box=BOX):
    sysm, pos = make(atoms)
    out = C.frame_contacts(sysm, pos, box)
    return {k: v for k, v in out.items() if v}


def ring(res, name, side, center, u, v, r=1.4):
    names = ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]
    u, v = np.array(u, float), np.array(v, float)
    return [(res, name, n, side, tuple(np.array(center) + r * (np.cos(a) * u + np.sin(a) * v)))
            for n, a in zip(names, np.linspace(0, 2 * np.pi, 6, endpoint=False))]


def amide(res, side, n_xyz, h_xyz):
    return [(res, "ALA", "N", side, n_xyz), (res, "ALA", "H", side, h_xyz)]


def test_backbone_hbond_needs_distance_and_angle():
    good = amide(0, 0, (0, 0, 0), (1.0, 0, 0)) + [(1, "GLY", "O", 1, (2.9, 0, 0))]
    assert found(good) == {"hbond_bb_bb": {(0, 1)}, "vdw": {(0, 1)}}
    far = amide(0, 0, (0, 0, 0), (1.0, 0, 0)) + [(1, "GLY", "O", 1, (3.7, 0, 0))]
    assert "hbond_bb_bb" not in found(far)
    bent = amide(0, 0, (0, 0, 0), (1.0, 0, 0)) + [(1, "GLY", "O", 1, (0.4, 2.9, 0))]
    assert "hbond_bb_bb" not in found(bent)


def test_sidechain_hbond_types():
    sc_bb = [(0, "LYS", "NZ", 0, (0, 0, 0)), (0, "LYS", "HZ1", 0, (1.0, 0, 0)),
             (1, "GLY", "O", 1, (2.9, 0, 0))]
    assert "hbond_sc_bb" in found(sc_bb)
    sc_sc = [(0, "SER", "OG", 0, (0, 0, 0)), (0, "SER", "HG", 0, (0.96, 0, 0)),
             (1, "ASN", "OD1", 1, (2.8, 0, 0))]
    assert "hbond_sc_sc" in found(sc_sc)


def test_salt_bridge_cutoff_and_donor_is_not_reversed():
    asp = (0, "ASP", "OD1", 0, (0, 0, 0))
    assert found([asp, (1, "LYS", "NZ", 1, (3.6, 0, 0))])["salt_bridge"] == {(0, 1)}
    assert "salt_bridge" not in found([asp, (1, "LYS", "NZ", 1, (4.4, 0, 0))])
    # binder anion, target cation: pair keeps (target, binder) order
    assert found([(0, "ARG", "NH1", 0, (0, 0, 0)), (1, "GLU", "OE1", 1, (3.2, 0, 0))])["salt_bridge"] == {(0, 1)}


def test_hydrophobic_is_a_subset_of_vdw_and_needs_apolar_residues():
    leu_val = [(0, "LEU", "CD1", 0, (0, 0, 0)), (1, "VAL", "CG1", 1, (3.8, 0, 0))]
    assert found(leu_val) == {"hydrophobic": {(0, 1)}, "vdw": {(0, 1)}}
    ser = [(0, "LEU", "CD1", 0, (0, 0, 0)), (1, "SER", "CB", 1, (3.8, 0, 0))]
    assert found(ser) == {"vdw": {(0, 1)}}
    assert found([(0, "LEU", "CD1", 0, (0, 0, 0)), (1, "VAL", "CG1", 1, (4.1, 0, 0))]) == {}


def test_pi_stacking_and_t_stacking_and_cation():
    x, y, z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    stack = ring(0, "PHE", 0, (0, 0, 0), x, y) + ring(1, "PHE", 1, (0, 0, 3.8), x, y)
    assert found(stack)["pi_stacking"] == {(0, 1)}
    apart = ring(0, "PHE", 0, (0, 0, 0), x, y) + ring(1, "PHE", 1, (0, 0, 7.5), x, y)
    assert "pi_stacking" not in found(apart)
    slid = ring(0, "PHE", 0, (0, 0, 0), x, y) + ring(1, "PHE", 1, (6.0, 0, 3.8), x, y)
    assert "pi_stacking" not in found(slid)              # centroid axis 57 deg off the normals
    tee = ring(0, "PHE", 0, (0, 0, 0), x, y) + ring(1, "PHE", 1, (0, 0, 4.8), y, z)
    got = found(tee)
    assert got["t_stacking"] == {(0, 1)} and "pi_stacking" not in got

    cat = ring(0, "PHE", 0, (0, 0, 0), x, y) + [(1, "LYS", "NZ", 1, (0, 0, 4.0))]
    assert found(cat)["pi_cation"] == {(0, 1)}
    edge = ring(0, "PHE", 0, (0, 0, 0), x, y) + [(1, "LYS", "NZ", 1, (4.0, 0, 0))]
    assert "pi_cation" not in found(edge)
    # ring on the binder, cation on the target: pair order stays (target, binder)
    rev = [(0, "LYS", "NZ", 0, (0, 0, 4.0))] + ring(1, "PHE", 1, (0, 0, 0), x, y)
    assert found(rev)["pi_cation"] == {(0, 1)}


def test_trp_uses_both_rings():
    trp = [(0, "TRP", n, 0, p) for n, p in zip(
        ["CG", "CD1", "NE1", "CE2", "CD2"],
        [(1.0, 0, 0), (0.3, 1.0, 0), (-0.8, 0.6, 0), (-0.8, -0.6, 0), (0.3, -1.0, 0)])]
    six = [(0, "TRP", n, 0, (-0.8 + 1.4 * np.cos(a), 1.4 * np.sin(a) - 0.0, 0.0))
           for n, a in zip(["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"], np.linspace(0, 2 * np.pi, 6, endpoint=False))]
    sysm, _ = make(trp + six + [(1, "LYS", "NZ", 1, (0, 0, 4))])
    assert len(sysm.rings) == 2


def test_periodic_images_are_handled():
    good = amide(0, 0, (0.5, 0, 0), (1.5, 0, 0)) + [(1, "GLY", "O", 1, (3.4, 0, 0))]
    moved = [a[:4] + ((a[4][0] + (BOX[0] if a[0] == 1 else 0), a[4][1], a[4][2]),) for a in good]
    assert "hbond_bb_bb" in found(moved)
    wrapped = [a[:4] + ((a[4][0] - 1.0 - (BOX[0] if a[0] == 0 else 0), a[4][1], a[4][2]),) for a in good]
    assert "hbond_bb_bb" in found(wrapped)


def test_same_group_atoms_never_contact():
    both_target = [(0, "LEU", "CD1", 0, (0, 0, 0)), (1, "VAL", "CG1", 0, (3.8, 0, 0))]
    assert found(both_target) == {}


def _state(xi, n, any_count, residues):
    return {"xi": xi, "n_frames": n, "counts": {C.ANY: any_count, "hbond_bb_bb": any_count / 2},
            "pairs": {}, "residues": residues}


def test_summary_finds_half_loss_hotspots_and_incomplete_pull():
    states = [_state(1.3 + 0.2 * i, 10, c, r) for i, (c, r) in enumerate([
        (12, {1: 1.0, 2: 1.0, 3: 0.9}), (10, {1: 1.0, 2: 1.0, 3: 0.4}), (5, {1: 1.0, 2: 0.2}),
        (1, {1: 0.6}), (0, {})])]
    s = C.summarize(states, bound_span_nm=0.25)
    assert s["xi_half"] == pytest.approx(1.3 + 0.2 * 2)
    assert s["hotspots"][0] == 1 and set(s["residue_persistence"]) == {1, 2, 3}
    assert s["residue_persistence"][1]["xi_last"] == pytest.approx(1.9)
    assert s["pull_complete"] and s["contacts_at_end"] == pytest.approx(0.5)

    still = states[:4]
    assert not C.summarize(still)["pull_complete"]


def test_accumulator_counts_frames_and_union():
    acc = C.Accumulator()
    a = {t: set() for t in C.TYPES}
    a["salt_bridge"] = {(1, 2)}
    a["vdw"] = {(1, 2), (3, 4)}
    acc.add(a, 1.3)
    acc.add({t: set() for t in C.TYPES}, 1.4)
    st = acc.state()
    assert st["n_frames"] == 2 and st["xi"] == pytest.approx(1.35)
    assert st["counts"][C.ANY] == 1.0 and st["pairs"][(1, 2, "salt_bridge")] == 0.5
    assert st["residues"][3] == 0.5


def test_labels_number_segments_only_for_multichain_groups():
    resids = np.array([1, 2, 3, 1, 2, 5])
    labels = C.residue_labels(["ALA"] * 6, resids, np.arange(6), np.array([0, 0, 0, 0, 0, 1]))
    assert labels[0] == "T1:ALA1" and labels[3] == "T2:ALA1" and labels[5] == "B:ALA5"


def test_consensus_needs_two_replicas():
    assert C.consensus([["a", "b"], ["b", "c"], ["b", "a"]]) == ["b", "a"]
    assert C.consensus([["a"]]) == ["a"]


mda = pytest.importorskip("MDAnalysis")


def _write_replica(tmp_path):
    """One replica dir: 2 tiny 'residues', a 2-window ladder that pulls a backbone hbond apart."""
    n = 4
    u = mda.Universe.empty(n, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    u.add_TopologyAttr("name", ["N", "H", "O", "C"])
    u.add_TopologyAttr("resname", ["ALA", "GLY"])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("segid", ["S"])
    coords = np.array([[10, 10, 10], [11, 10, 10], [12.9, 10, 10], [13.5, 11, 10]], float)
    u.atoms.positions = coords
    u.dimensions = [40, 40, 40, 90, 90, 90]
    u.atoms.write(str(tmp_path / "em.gro"))
    (tmp_path / "index.ndx").write_text("[ Target ]\n1 2\n[ Binder ]\n3 4\n")
    for w, shift in enumerate((0.0, 6.0)):                    # window 1: binder 6 A further
        moved = coords.copy()
        moved[2:, 0] += shift
        with mda.Writer(str(tmp_path / f"umbrella_win{w}_conf{w}.xtc"), n) as wr:
            for i in range(6):
                u.atoms.positions = moved
                u.trajectory.ts.time = 250.0 * i
                u.trajectory.ts.dimensions = [40, 40, 40, 90, 90, 90]
                wr.write(u.atoms)
        xi = 0.3 + 0.6 * w
        (tmp_path / f"umbrella_win{w}_conf{w}_pullx.xvg").write_text(
            "@ title \"Pull COM\"\n" + "".join(f"{250.0 * i}\t{xi}\n" for i in range(6)))
    (tmp_path / "pullx_files.dat").write_text("umbrella_win0_conf0_pullx.xvg\numbrella_win1_conf1_pullx.xvg\n")


def test_windows_pipeline_end_to_end(tmp_path):
    from cg_us.config import Protocol
    _write_replica(tmp_path)
    proto = Protocol()
    res = C.analyse_replica(tmp_path, proto, sources=("windows",))
    s = res["windows"]
    assert s["bound_counts"]["hbond_bb_bb"] == pytest.approx(1.0)
    assert s["xi_bound"] == pytest.approx(0.3)
    assert s["pull_complete"] is True                       # window 1 has no contacts left
    files = {p.name for p in (tmp_path / "analysis").iterdir()}
    assert {"contacts_windows_pairs.csv", "contacts_windows_counts.csv", "contacts_windows.json"} <= files
    assert json.loads((tmp_path / "analysis" / "contacts_windows.json").read_text())["hotspots"]
    import pandas as pd
    counts = pd.read_csv(tmp_path / "analysis" / "contacts_windows_counts.csv")
    assert list(counts["xi_nm"]) == [0.3, 0.9] and counts["hbond_bb_bb"].tolist() == [1.0, 0.0]
    # first 500 ps of each window are discarded: 6 frames -> 4 kept
    assert list(counts["n_frames"]) == [4, 4]


def test_missing_mdanalysis_says_how_to_install(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "MDAnalysis", None)
    with pytest.raises(RuntimeError, match="pip install"):
        C._import_mda()
