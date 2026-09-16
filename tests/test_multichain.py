"""Pull groups that span more than one chain.

The two-chain case (one target chain, one binder chain) is covered by
test_units; everything here is about a group made of several chains: a receptor
dimer pulled against a peptide, or a binder grafted onto two strands.
"""

from pathlib import Path

import numpy as np
import pytest

from cg_us import structure, topology
from cg_us.backends.base import GmxError, _check_group_fits_box, _restrain_group
from cg_us.manifest import Entry
from cg_us.prep import _chain_order


def _atoms(n: int, resname: str = "ALA") -> str:
    return "\n".join(
        f"{i:6d}   CT  {i}  {resname}  CA  {i}   0.0  12.0" for i in range(1, n + 1)
    )


def _three_chain_topology(tmp_path: Path, split: bool = False) -> Path:
    """Target = chains M + N (4 and 3 atoms), binder = chain P (2 atoms)."""
    blocks = {"M": 4, "N": 3, "P": 2}
    top = tmp_path / "topol.top"
    if not split:
        body = "".join(
            f"[ moleculetype ]\nProtein_chain_{ch}     3\n\n[ atoms ]\n{_atoms(n)}\n\n"
            for ch, n in blocks.items()
        )
        top.write_text(body + "[ system ]\nx\n\n[ molecules ]\n"
                       "Protein_chain_M 1\nProtein_chain_N 1\nProtein_chain_P 1\nSOL 5\n")
        return top
    includes = ""
    for ch, n in blocks.items():
        (tmp_path / f"topol_Protein_chain_{ch}.itp").write_text(
            f"[ moleculetype ]\nProtein_chain_{ch}     3\n\n[ atoms ]\n{_atoms(n)}\n")
        includes += f'#include "topol_Protein_chain_{ch}.itp"\n'
    top.write_text(includes + "\n[ system ]\nx\n\n[ molecules ]\n"
                   "Protein_chain_M 1\nProtein_chain_N 1\nProtein_chain_P 1\nSOL 5\n")
    return top


def _atom(chain: str, i: int, xyz) -> structure.Atom:
    return structure.Atom("ATOM", i, "CA", "ALA", chain, i, "",
                          np.array(xyz, dtype=float), "C", "")


def _report(cyclic: bool) -> structure.ChainReport:
    return structure.ChainReport("_", 1, 1, 1, 1, 0.13, cyclic, [], [])


# --- chain specs ------------------------------------------------------------

def test_parse_chains_reads_group_specs():
    assert structure.parse_chains("A") == ["A"]
    assert structure.parse_chains("M+N") == ["M", "N"]
    assert structure.parse_chains("M, N") == ["M", "N"]
    assert structure.parse_chains("M/N P") == ["M", "N", "P"]
    assert structure.parse_chains(["M", "N"]) == ["M", "N"]


def test_parse_chains_rejects_repeats_and_emptiness():
    with pytest.raises(ValueError, match="listed twice"):
        structure.parse_chains("M+M")
    with pytest.raises(ValueError, match="no chain ID"):
        structure.parse_chains("  ")


def test_entry_exposes_the_two_groups():
    e = Entry(name="x", pdb=Path("x.pdb"), target="M+N", binder="P")
    assert e.target_chains == ["M", "N"]
    assert e.binder_chains == ["P"]
    assert e.all_chains == ["M", "N", "P"]
    assert e.multi_chain
    assert not Entry(name="y", pdb=Path("y.pdb"), target="A", binder="B").multi_chain


def test_group_selection_spans_chains(tmp_path):
    atoms = [_atom("M", 1, [0, 0, 0]), _atom("N", 2, [1, 0, 0]), _atom("P", 3, [2, 0, 0])]
    assert len(structure.select_chains(atoms, "M+N")) == 2
    with pytest.raises(ValueError, match="not present"):
        structure.select_chains(atoms, "M+Z")


# --- topology ---------------------------------------------------------------

def test_multichain_group_resolves_to_one_moltype_per_chain(tmp_path):
    top = _three_chain_topology(tmp_path)
    assert topology.chain_moltype_groups(top, "M+N", "P") == {
        "Target": ["Protein_chain_M", "Protein_chain_N"],
        "Binder": ["Protein_chain_P"],
    }


def test_multichain_pull_group_accumulates_every_chain(tmp_path):
    top = _three_chain_topology(tmp_path)
    groups = topology.pull_group_indices(top, "M+N", "P")
    assert groups["Target"] == list(range(1, 8))    # 4 atoms of M + 3 of N
    assert groups["Binder"] == [8, 9]


def test_chain_order_of_the_spec_decides_which_atoms_come_first(tmp_path):
    """The group is a set of atoms, but the moltype list follows the spec order."""
    top = _three_chain_topology(tmp_path)
    assert topology.chain_moltype_groups(top, "N+M", "P")["Target"] == [
        "Protein_chain_N", "Protein_chain_M"]
    # indices are gathered in [ molecules ] order either way
    assert topology.pull_group_indices(top, "N+M", "P")["Target"] == list(range(1, 8))


def test_flat_chain_moltypes_refuses_a_multichain_group(tmp_path):
    top = _three_chain_topology(tmp_path)
    with pytest.raises(ValueError, match="spans 2 chains"):
        topology.chain_moltypes(top, "M+N", "P")
    assert topology.chain_moltypes(top, "M", "P") == {
        "Target": "Protein_chain_M", "Binder": "Protein_chain_P"}


def test_order_fallback_maps_all_chains_positionally(tmp_path):
    """Chain letters lost: the n-th protein moltype is the n-th chain written."""
    (tmp_path / "chainless.itp").write_text(
        f"[ moleculetype ]\nProteinA     3\n\n[ atoms ]\n{_atoms(4)}\n\n"
        f"[ moleculetype ]\nProteinB     3\n\n[ atoms ]\n{_atoms(3)}\n\n"
        f"[ moleculetype ]\nProteinC     3\n\n[ atoms ]\n{_atoms(2)}\n")
    top = tmp_path / "topol.top"
    top.write_text('#include "chainless.itp"\n\n[ system ]\nx\n\n[ molecules ]\n'
                   "ProteinA 1\nProteinB 1\nProteinC 1\n")
    groups = topology.chain_moltype_groups(top, "M+N", "P", order=["P", "M", "N"])
    assert groups == {"Target": ["ProteinB", "ProteinC"], "Binder": ["ProteinA"]}
    idx = topology.pull_group_indices(top, "M+N", "P", order=["P", "M", "N"])
    assert idx["Binder"] == list(range(1, 5))
    assert idx["Target"] == list(range(5, 10))


# --- prep -------------------------------------------------------------------

def test_cyclic_chain_leads_the_multichain_order():
    entry = Entry(name="x", pdb=Path("x.pdb"), target="M+N", binder="P")
    linear = {ch: _report(False) for ch in "MNP"}
    assert _chain_order(entry, linear) == ["M", "N", "P"]
    assert _chain_order(entry, {**linear, "P": _report(True)}) == ["P", "M", "N"]
    assert _chain_order(entry, {**linear, "N": _report(True)}) == ["N", "M", "P"]


def test_two_cyclic_chains_are_refused_instead_of_built_linear():
    entry = Entry(name="x", pdb=Path("x.pdb"), target="M+N", binder="P")
    reports = {"M": _report(True), "N": _report(False), "P": _report(True)}
    with pytest.raises(ValueError, match="first chain only"):
        _chain_order(entry, reports)


def test_orientation_uses_the_whole_target_group():
    atoms = [_atom("M", 1, [0, 0, 0]), _atom("M", 2, [0, 10, 0]),
             _atom("N", 3, [0, 0, 10]), _atom("N", 4, [0, 10, 10]),
             _atom("P", 5, [20, 5, 5])]
    oriented, info = structure.orient_for_pull(atoms, "M+N", "P")
    axis = (structure.com(structure.select_chains(oriented, "P"))
            - structure.com(structure.select_chains(oriented, "M+N")))
    assert axis[2] > 0
    assert np.allclose(axis[:2], 0, atol=1e-6)
    assert info["com_distance_nm"] == pytest.approx(2.0, abs=1e-6)
    assert info["target_extent_nm"] == pytest.approx([0.0, 1.0, 1.0], abs=1e-6)
    assert info["binder_extent_nm"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


# --- restraints and PBC -----------------------------------------------------

class _Ctx:
    def __init__(self, workdir):
        self.workdir = workdir


def test_every_chain_of_a_restrained_group_gets_pinned(tmp_path):
    top = _three_chain_topology(tmp_path, split=True)
    added = _restrain_group(_Ctx(tmp_path), top, ["Protein_chain_M", "Protein_chain_N"],
                            "target", "POSRES_TARGET", "backbone", None)
    assert len(added) == 2
    for ch in ("M", "N"):
        assert (tmp_path / f"posre_target_Protein_chain_{ch}.itp").exists()
        assert "POSRES_TARGET" in (tmp_path / f"topol_Protein_chain_{ch}.itp").read_text()
    assert "POSRES_TARGET" not in (tmp_path / "topol_Protein_chain_P.itp").read_text()


def test_single_chain_group_keeps_the_old_filename(tmp_path):
    top = _three_chain_topology(tmp_path, split=True)
    _restrain_group(_Ctx(tmp_path), top, ["Protein_chain_P"], "binder",
                    "POSRES_BINDER", "ca", 200.0)
    assert (tmp_path / "posre_binder.itp").exists()


def test_group_wider_than_half_the_box_is_refused():
    box = np.array([4.0, 4.0, 8.0])
    xyz = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    with pytest.raises(GmxError, match="past half the box"):
        _check_group_fits_box("Target", xyz, xyz[0], box)


def test_group_inside_half_the_box_passes():
    box = np.array([6.0, 6.0, 12.0])
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 3.0], [-2.0, -1.0, -3.0]])
    _check_group_fits_box("Target", xyz, xyz[0], box)
