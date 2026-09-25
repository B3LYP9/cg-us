"""`--enqueue`: the command goes to a ClearML queue instead of running, and the
launcher the agent executes runs it faithfully. ClearML itself is faked."""

import json
import runpy
import sys
import types
from pathlib import Path

import pytest

from cg_us import clearml_queue as cq
from cg_us import cli


class FakeTask:
    created = []
    enqueued = []
    current = None

    def __init__(self, project, name):
        self.project, self.name, self.id = project, name, f"task{len(FakeTask.created) + 1}"
        self.params, self.tags = {}, []

    @classmethod
    def create(cls, project_name=None, task_name=None, **kw):
        t = cls(project_name, task_name)
        t.kw = kw
        cls.created.append(t)
        return t

    @classmethod
    def enqueue(cls, task, queue_name=None):
        cls.enqueued.append((task.id, queue_name))

    @classmethod
    def current_task(cls):
        return cls.current

    @classmethod
    def init(cls, **kw):
        return cls(kw.get("project_name"), kw.get("task_name"))

    def set_parameters(self, d):
        self.params.update(d)

    def set_parameter(self, k, v):
        self.params[k] = v

    def add_tags(self, tags):
        self.tags += tags

    def get_output_log_web_page(self):
        return f"http://clearml/{self.id}"

    def get_parameters_as_dict(self):
        out = {}
        for k, v in self.params.items():
            sec, _, key = k.partition("/")
            out.setdefault(sec, {})[key] = v
        return out


@pytest.fixture
def fake_clearml(monkeypatch):
    FakeTask.created, FakeTask.enqueued, FakeTask.current = [], [], None
    mod = types.ModuleType("clearml")
    mod.Task = FakeTask
    monkeypatch.setitem(sys.modules, "clearml", mod)
    monkeypatch.delenv("CGUS_CLEARML_QUEUE", raising=False)
    return FakeTask


def test_split_own_flags_strips_queue_options_only():
    argv = ["run", "--root", "r", "--system", "a", "b", "--queue=q2", "--task-name", "night", "--enqueue"]
    child, own = cq.split_own_flags(argv)
    assert child == ["run", "--root", "r", "--system", "a", "b"]
    assert own == {"queue": "q2", "name": "night", "project": None, "enqueue": True}
    assert cq.split_own_flags(["run", "--project=GP1"])[1] == {
        "queue": None, "name": None, "project": "GP1", "enqueue": True}

    child, own = cq.split_own_flags(["analyze", "--root", "r"])
    assert child == ["analyze", "--root", "r"] and not own["enqueue"]

    # --queue alone already means "enqueue"
    assert cq.split_own_flags(["run", "--queue", "a100-2"])[1]["enqueue"]
    with pytest.raises(cq.QueueError):
        cq.split_own_flags(["run", "--queue"])


def test_enqueue_creates_one_task_on_the_default_queue(fake_clearml, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "/x/bin:/usr/bin")
    boom = lambda a: pytest.fail("the command must not run when it is only enqueued")
    monkeypatch.setattr(cli, "cmd_run", boom)

    rc = cli.main(["run", "--root", "runs/gdf8", "--system", "s1", "s2", "s3", "--enqueue"])
    assert rc == 0

    (task,) = fake_clearml.created
    assert fake_clearml.enqueued == [(task.id, "a100-1")]
    assert task.name == "cg-us run gdf8 s1,s2 +1"
    assert task.project == "GP20181/Umbrella sampling/run"
    assert task.kw["script"].endswith("_clearml_launcher.py")
    assert task.kw["force_single_script_file"] is True
    p = task.get_parameters_as_dict()["cgus"]
    assert json.loads(p["argv"]) == ["run", "--root", "runs/gdf8", "--system", "s1", "s2", "s3"]
    assert p["cwd"] == str(tmp_path.resolve())
    assert json.loads(p["env"])["PATH"] == "/x/bin:/usr/bin"
    assert json.loads(p["exe"])[0] == sys.executable
    assert "enqueued on 'a100-1'" in capsys.readouterr().out


def test_queue_can_be_chosen_by_flag_and_by_environment(fake_clearml, monkeypatch):
    assert cli.main(["analyze", "--root", "r", "--queue", "a100-2"]) == 0
    monkeypatch.setenv("CGUS_CLEARML_QUEUE", "h100")
    assert cli.main(["extend", "--root", "r", "--fill-gaps", "--enqueue"]) == 0
    assert [q for _, q in fake_clearml.enqueued] == ["a100-2", "h100"]
    assert "--fill-gaps" in json.loads(fake_clearml.created[1].get_parameters_as_dict()["cgus"]["argv"])


def test_two_submissions_keep_their_order(fake_clearml):
    cli.main(["run", "--root", "r", "--enqueue"])
    cli.main(["analyze", "--root", "r", "--enqueue"])
    cli.main(["extend", "--root", "r", "--enqueue"])
    assert [t.name.split()[1] for t in fake_clearml.created] == ["run", "analyze", "extend"]
    assert [tid for tid, _ in fake_clearml.enqueued] == [t.id for t in fake_clearml.created]


def test_missing_clearml_says_how_to_install(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "clearml", None)  # import raises ImportError
    assert cli.main(["run", "--root", "r", "--enqueue"]) == 1
    assert "pip install" in capsys.readouterr().err


def test_queue_status_is_not_mistaken_for_an_enqueue(monkeypatch, fake_clearml):
    seen = {}
    monkeypatch.setattr(cq, "queue_status", lambda q=None: seen.update(q=q) or [])
    assert cli.main(["queue", "status", "--queue", "v100"]) == 0
    assert seen["q"] == "v100" and not fake_clearml.created


LAUNCHER = Path(cq.__file__).with_name("_clearml_launcher.py")


def _run_launcher(fake_clearml, tmp_path, script, argv_extra=()):
    task = FakeTask("cg-us", "job")
    task.params = {
        "cgus/exe": json.dumps([sys.executable, "-c", script]),
        "cgus/argv": json.dumps(list(argv_extra)),
        "cgus/cwd": str(tmp_path),
        "cgus/env": json.dumps({"CGUS_TEST": "forwarded"}),
    }
    fake_clearml.current = task
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(LAUNCHER), run_name="__main__")
    return task, exc.value.code


def test_launcher_runs_in_the_recorded_cwd_with_the_recorded_env(fake_clearml, tmp_path, capsys):
    script = "import os,sys; print(os.getcwd()); print(os.environ['CGUS_TEST']); print(sys.argv[1:])"
    task, code = _run_launcher(fake_clearml, tmp_path, script, ["run", "--root", "x"])
    out = capsys.readouterr().out
    assert code == 0
    assert str(tmp_path.resolve()) in out or str(tmp_path) in out
    assert "forwarded" in out and "['run', '--root', 'x']" in out
    assert task.params["cgus/exit_code"] == "0"


def test_launcher_passes_the_exit_code_through_so_the_task_fails(fake_clearml, tmp_path):
    task, code = _run_launcher(fake_clearml, tmp_path, "import sys; print('boom'); sys.exit(3)")
    assert code == 3 and task.params["cgus/exit_code"] == "3"


def test_project_is_an_argument_and_commands_go_to_their_own_subfolder(fake_clearml, monkeypatch):
    cli.main(["run", "--root", "r", "--enqueue"])
    cli.main(["analyze", "--root", "r", "--project", "GP99999"])
    cli.main(["extend", "--root", "r", "--project=GP99999", "--fill-gaps"])
    cli.main(["prep", "--root", "r", "--enqueue"])
    assert [t.project for t in fake_clearml.created] == [
        "GP20181/Umbrella sampling/run",
        "GP99999/Umbrella sampling/analysis",
        "GP99999/Umbrella sampling/extend",
        "GP20181/Umbrella sampling/run",
    ]
    # the project flag is for the queue, not for the command that runs later
    assert "--project" not in json.loads(fake_clearml.created[1].get_parameters_as_dict()["cgus"]["argv"])
    monkeypatch.setenv("CGUS_CLEARML_PROJECT", "GP7")
    cli.main(["run", "--root", "r", "--enqueue"])
    assert fake_clearml.created[-1].project == "GP7/Umbrella sampling/run"
