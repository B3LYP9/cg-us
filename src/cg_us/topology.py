"""topol.top inspection and position-restraint wiring.

Atom ordering produced by pdb2gmx survives solvate/genion, so global index
ranges for the pull groups can be read straight out of the topology instead of
being guessed from the .gro file (which carries no chain information).

For multi-chain input pdb2gmx does not inline the ``[ moleculetype ]`` blocks:
topol.top only ``#include``s per-chain ``topol_Protein_chain_X.itp`` files. The
parser therefore follows local includes, remembers which file each molecule type
came from, and writes restraint hooks back into that file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

MOLTYPE_RE = re.compile(r"^\s*\[\s*moleculetype\s*\]\s*$")
SECTION_RE = re.compile(r"^\s*\[\s*(\w+)\s*\]\s*$")
INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"')


@dataclass
class MolType:
    name: str
    atoms: list[tuple[int, str, int, str]] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    source: Path | None = None
    header_line: int = 0

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def chain(self) -> str | None:
        m = re.search(r"chain_([A-Za-z0-9])(?:\d*)?$", self.name)
        return m.group(1) if m else None

    @property
    def is_protein(self) -> bool:
        return self.name.lower().startswith(("protein", "peptide"))


def _walk(path: Path, seen: set[Path]) -> Iterator[tuple[str, Path, int]]:
    """Yield (line, file, line_number) following local #include directives."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for i, raw in enumerate(lines):
        inc = INCLUDE_RE.match(raw)
        if inc:
            target = inc.group(1)
            name = Path(target).name
            if ".ff/" in target or target.startswith("/") or name.startswith("posre"):
                continue
            child = (path.parent / target).resolve()
            if child.exists() and child not in seen:
                seen.add(child)
                yield from _walk(child, seen)
            continue
        yield raw, path, i


def parse_topology(path: str | Path) -> tuple[list[MolType], list[tuple[str, int]]]:
    path = Path(path)
    moltypes: list[MolType] = []
    molecules: list[tuple[str, int]] = []
    section = None
    current: MolType | None = None

    for raw, source, lineno in _walk(path, {path.resolve()}):
        line = raw.split(";")[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        m = SECTION_RE.match(line)
        if m:
            section = m.group(1).lower()
            if section == "moleculetype":
                current = None
            continue
        if section == "moleculetype" and current is None:
            current = MolType(name=line.split()[0], source=source, header_line=lineno)
            moltypes.append(current)
        elif section == "atoms" and current is not None:
            f = line.split()
            if len(f) >= 5 and f[0].isdigit():
                current.atoms.append((int(f[0]), f[4], int(f[2]), f[3]))
        elif section == "bonds" and current is not None:
            f = line.split()
            if len(f) >= 2 and f[0].isdigit() and f[1].isdigit():
                current.bonds.append((int(f[0]), int(f[1])))
        elif section == "molecules":
            f = line.split()
            if len(f) >= 2:
                molecules.append((f[0], int(f[1])))
    return moltypes, molecules


def is_cyclic(mt: MolType) -> bool:
    """True when the backbone ring is actually closed in the topology.

    pdb2gmx only cyclises the *first* chain it processes (GROMACS issue 5091),
    and it fails silently: the ring bond is dropped and charged termini are
    added instead, which puts an NH3+ nitrogen on top of a carboxylate. The
    resulting infinite force is the real reason those systems died in EM, so
    the topology is checked rather than trusted.
    """
    if not mt.atoms or not mt.bonds:
        return False
    first_res = mt.atoms[0][2]
    last_res = mt.atoms[-1][2]
    if first_res == last_res:
        return False
    n = next((a[0] for a in mt.atoms if a[2] == first_res and a[1] == "N"), None)
    c = next((a[0] for a in reversed(mt.atoms) if a[2] == last_res and a[1] == "C"), None)
    if n is None or c is None:
        return False
    return (n, c) in mt.bonds or (c, n) in mt.bonds


def moltype(top: str | Path, name: str) -> MolType | None:
    return next((m for m in parse_topology(top)[0] if m.name == name), None)


def _resolve_chains(moltypes: list[MolType], molecules: list[tuple[str, int]],
                    target_chain: str, binder_chain: str,
                    order: list[str] | None = None) -> dict[str, str]:
    """Map Target/Binder onto molecule type names.

    Preferred route is the ``chain_X`` suffix pdb2gmx writes. If chain letters
    were lost (single-chain naming, merged chains, renamed molecule types), fall
    back on order: cg-us writes the input PDB with the target chain first, so
    the first protein molecule type in [ molecules ] is the target.
    """
    by_chain = {m.chain: m.name for m in moltypes if m.chain}
    if target_chain in by_chain and binder_chain in by_chain:
        return {"Target": by_chain[target_chain], "Binder": by_chain[binder_chain]}

    by_name = {m.name: m for m in moltypes}
    ordered = [name for name, _ in molecules
               if name in by_name and by_name[name].is_protein]
    unique = list(dict.fromkeys(ordered))
    if len(unique) >= 2:
        written = order or [target_chain, binder_chain]
        first = "Target" if written[0] == target_chain else "Binder"
        second = "Binder" if first == "Target" else "Target"
        return {first: unique[0], second: unique[1]}

    found = ", ".join(m.name for m in moltypes) or "none"
    raise ValueError(
        f"cannot map chains {target_chain}/{binder_chain} onto molecule types "
        f"(found: {found}). Expected pdb2gmx to emit one protein molecule type "
        "per chain; check that the input PDB kept its chain IDs and TER records."
    )


def chain_moltypes(top: str | Path, target_chain: str, binder_chain: str,
                   order: list[str] | None = None) -> dict[str, str]:
    """{'Target': <moleculetype>, 'Binder': <moleculetype>}."""
    moltypes, molecules = parse_topology(top)
    return _resolve_chains(moltypes, molecules, target_chain, binder_chain, order)


def pull_group_indices(top: str | Path, target_chain: str, binder_chain: str,
                       order: list[str] | None = None) -> dict[str, list[int]]:
    moltypes, molecules = parse_topology(top)
    mapping = _resolve_chains(moltypes, molecules, target_chain, binder_chain, order)
    wanted = {name: label for label, name in mapping.items()}
    by_name = {m.name: m for m in moltypes}

    offset = 0
    groups: dict[str, list[int]] = {}
    for name, count in molecules:
        mt = by_name.get(name)
        n = mt.n_atoms if mt else 0
        for _ in range(count):
            label = wanted.get(name)
            if label:
                groups.setdefault(label, []).extend(range(offset + 1, offset + n + 1))
            offset += n

    missing = {"Target", "Binder"} - groups.keys()
    if missing:
        raise ValueError(f"molecule types resolved but absent from [ molecules ]: {missing}")
    return groups


def append_index_groups(index: str | Path, groups: dict[str, list[int]]) -> None:
    index = Path(index)
    text = index.read_text() if index.exists() else ""
    chunks = [text.rstrip("\n")] if text.strip() else []
    for name, idx in groups.items():
        body = "\n".join(
            " ".join(f"{v:5d}" for v in idx[i:i + 15]) for i in range(0, len(idx), 15)
        )
        chunks.append(f"[ {name} ]\n{body}")
    index.write_text("\n".join(chunks) + "\n")


def read_index_groups(index: str | Path) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    name = None
    for line in Path(index).read_text().splitlines():
        head = re.match(r"\s*\[\s*(.+?)\s*\]", line)
        if head:
            name = head.group(1)
            groups[name] = []
        elif name:
            groups[name].extend(int(v) for v in line.split())
    return groups


def restraint_itp(top: str | Path, moltype: str, out: str | Path,
                  selection: str = "backbone", k: float = 1000.0) -> Path:
    moltypes, _ = parse_topology(top)
    mt = next((m for m in moltypes if m.name == moltype), None)
    if mt is None:
        raise ValueError(f"moleculetype '{moltype}' not found in {top}")

    keep = {"backbone": {"N", "CA", "C", "O"}, "ca": {"CA"}}.get(selection)
    lines = ["[ position_restraints ]", "; generated by cg-us",
             ";  i funct       fcx        fcy        fcz"]
    for serial, name, _, _ in mt.atoms:
        if keep is not None and name not in keep:
            continue
        if keep is None and name.startswith("H"):
            continue
        lines.append(f"{serial:6d}     1 {k:10.1f} {k:10.1f} {k:10.1f}")
    out = Path(out)
    out.write_text("\n".join(lines) + "\n")
    return out


def wire_restraints(top: str | Path, moltype: str, itp_name: str, define: str) -> Path:
    """Insert an ``#ifdef <define>`` include at the end of a moleculetype block.

    The block is written into whichever file actually declares the molecule
    type, which for multi-chain systems is topol_Protein_chain_X.itp rather than
    topol.top.
    """
    moltypes, _ = parse_topology(top)
    mt = next((m for m in moltypes if m.name == moltype), None)
    if mt is None or mt.source is None:
        raise ValueError(f"moleculetype '{moltype}' not found in {top}")

    source = Path(mt.source)
    lines = source.read_text().splitlines()
    if any(f'#include "{itp_name}"' in l for l in lines):
        return source

    starts = [i for i, l in enumerate(lines) if MOLTYPE_RE.match(l)]
    later = [i for i in starts if i > mt.header_line]
    end = later[0] if later else None
    if end is None:
        end = next((i for i, l in enumerate(lines)
                    if i > mt.header_line and SECTION_RE.match(l)
                    and SECTION_RE.match(l).group(1).lower() == "system"), len(lines))

    block = ["", f"; immobile reference for umbrella sampling - added by cg-us",
             f"#ifdef {define}", f'#include "{itp_name}"', "#endif", ""]
    lines[end:end] = block
    source.write_text("\n".join(lines) + "\n")
    return source


def target_moltype(top: str | Path, chain: str) -> str:
    moltypes, molecules = parse_topology(top)
    for m in moltypes:
        if m.chain == chain:
            return m.name
    protein = [m for m in moltypes if m.is_protein]
    if len(protein) == 1:
        return protein[0].name
    found = ", ".join(m.name for m in moltypes) or "none"
    raise ValueError(f"no moleculetype for chain '{chain}' in {top} (found: {found})")
