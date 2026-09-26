"""prep --add-replicas adds replicas to a prepared root without touching the old ones."""

import json
from pathlib import Path

import pytest

from cg_us import cli, prep
from cg_us.config import Protocol
from cg_us.manifest import read_manifest

BENCH = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def root(tmp_path):
    manifest = BENCH / "manifest.csv"
    if not manifest.exists():
        pytest.skip("example manifest not available")
    r = tmp_path / "runs"
    proto = Protocol()
    proto.replicas = 2
    r.mkdir()
    proto.dump(r / "protocol.yaml")
    entries = read_manifest(manifest, BENCH)
    for e in entries:
        prep.prepare_entry(e, proto, r)
    (r / "run_state.json").write_text(json.dumps({"manifest": str(manifest), "manifest_root": str(BENCH),
                                                   "systems": [e.name for e in entries]}))
    return r, manifest, entries


def test_add_replicas_keeps_old_replicas_and_extends_the_protocol(root):
    r, manifest, entries = root
    name = entries[0].name
    old_mdp = r / "systems" / name / "rep1" / "md_umbrella.mdp"
    old_mdp.write_text(old_mdp.read_text() + "\n; edited by a run\n")        # a run's own changes must survive
    before = old_mdp.read_text()
    state = (r / "run_state.json").read_text()

    rc = cli.main(["prep", "--manifest", str(manifest), "--data-root", str(BENCH), "--root", str(r),
                   "--system", name, "--add-replicas", "3"])
    assert rc == 0
    sysdir = r / "systems" / name
    assert prep.existing_replicas(r, entries[0]) == [1, 2, 3, 4, 5]
    assert old_mdp.read_text() == before
    assert (r / "run_state.json").read_text() == state
    assert Protocol.load(r / "protocol.yaml").replicas == 5
    seeds = [x["seed"] for x in json.loads((sysdir / "prep.json").read_text())["replicas"]]
    assert len(seeds) == 5 and len(set(seeds)) == 5                       # independent velocity seeds
    other = entries[1]
    assert prep.existing_replicas(r, other) == [1, 2]                     # systems not named are left alone


def test_add_replicas_needs_a_prepared_system(root, capsys):
    r, manifest, entries = root
    rc = cli.main(["prep", "--manifest", str(manifest), "--data-root", str(BENCH), "--root", str(r / "elsewhere"),
                   "--add-replicas", "1"])
    assert rc == 1 and "protocol.yaml" in capsys.readouterr().err
