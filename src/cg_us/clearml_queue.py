"""Run cg-us commands through a ClearML queue instead of straight from the shell.

`cg-us run|analyze|extend|all|prep|bench ... --enqueue` does not do the work: it
creates a ClearML task that will run the very same command later, on whichever
clearml-agent serves the queue. The agent takes one task at a time, so commands
queued this way start one after another instead of fighting for the GPU and
the cores, and the queue shows up in the ClearML web UI with its console log.

The task is a tiny launcher (`_clearml_launcher.py`, uploaded with the task as a
single script) that only needs `clearml`. It does not rebuild an environment: it
runs the cg-us that submitted the job - same interpreter, same working directory,
same PATH - as a subprocess and forwards its output to the task console. The
agent daemon is not reconfigured, so other users of the queue are not affected.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_QUEUE = "a100-1"
DEFAULT_PROJECT = "GP20181"
# every queued task lives in <project>/<FOLDER>/<subfolder of its command>
FOLDER = "Umbrella sampling"
SUBFOLDER = {"run": "run", "prep": "run", "all": "run", "bench": "run",
             "analyze": "analysis", "extend": "extend"}
LAUNCHER = Path(__file__).with_name("_clearml_launcher.py")
PARAM_SECTION = "cgus"
QUEUEABLE = ("prep", "run", "extend", "analyze", "all", "bench")

# flags consumed here, never passed on to the command the task runs
_OWN_FLAGS_WITH_VALUE = ("--queue", "--task-name", "--project")
_OWN_FLAGS = ("--enqueue",)

# environment the queued command needs; the agent's own environment has neither
_FORWARDED_ENV = ("PATH", "CHAPERONg_PATH", "GMXLIB", "GMXBIN", "LD_LIBRARY_PATH")


class QueueError(RuntimeError):
    pass


def default_queue() -> str:
    return os.environ.get("CGUS_CLEARML_QUEUE", DEFAULT_QUEUE)


def default_project() -> str:
    return os.environ.get("CGUS_CLEARML_PROJECT", DEFAULT_PROJECT)


def project_path(argv: list[str], project: str | None = None) -> str:
    """ClearML project the task goes to: `<project>/Umbrella sampling/<run|analysis|extend>`."""
    sub = SUBFOLDER.get(argv[0] if argv else "", "run")
    return f"{project or default_project()}/{FOLDER}/{sub}"


def split_own_flags(argv: list[str]) -> tuple[list[str], dict]:
    """(argv without the queue flags, {"queue":..., "name":..., "project":..., "enqueue": bool}).

    Handles both `--queue q` and `--queue=q`. `--queue` or `--project` on their own
    imply `--enqueue`, so `cg-us run --queue a100-2 ...` is enough.
    """
    out: list[str] = []
    found = {"queue": None, "name": None, "project": None, "enqueue": False}
    i = 0
    while i < len(argv):
        tok = argv[i]
        name, eq, val = tok.partition("=")
        if name in _OWN_FLAGS_WITH_VALUE:
            if not eq:
                i += 1
                if i >= len(argv):
                    raise QueueError(f"{name} needs a value")
                val = argv[i]
            found[{"--queue": "queue", "--task-name": "name", "--project": "project"}[name]] = val
            if name in ("--queue", "--project"):
                found["enqueue"] = True
        elif tok in _OWN_FLAGS:
            found["enqueue"] = True
        else:
            out.append(tok)
        i += 1
    return out, found


def _import_clearml():
    try:
        import clearml  # noqa: F401
        from clearml import Task
    except ImportError as exc:
        raise QueueError(
            "clearml is not installed in this environment; run "
            "`pip install 'cg-us[queue]'` (or `pip install clearml`) to queue jobs"
        ) from exc
    return Task


def _describe(argv: list[str]) -> str:
    """Task title: `us_<command>_<run folder>`. The launcher appends
    `_<system>_rep<N>` while the command works on that replica."""
    cmd = argv[0] if argv else "cmd"
    root = None
    if "--root" in argv and argv.index("--root") + 1 < len(argv):
        root = Path(argv[argv.index("--root") + 1]).name
    return f"us_{cmd}_{root}" if root else f"us_{cmd}"


def _systems(argv: list[str]) -> list[str]:
    out: list[str] = []
    if "--system" in argv:
        j = argv.index("--system") + 1
        while j < len(argv) and not argv[j].startswith("--"):
            out.append(argv[j])
            j += 1
    return out


def _capture_env() -> dict:
    env = {k: os.environ[k] for k in _FORWARDED_ENV if k in os.environ}
    return env


def submit(argv: list[str], *, queue: str | None = None, name: str | None = None,
           project: str | None = None, cwd: str | None = None) -> dict:
    """Create and enqueue the ClearML task that will run `cg-us <argv>`."""
    Task = _import_clearml()
    queue = queue or default_queue()
    cwd = str(Path(cwd or os.getcwd()).resolve())
    name = name or _describe(argv)
    project_name = project_path(argv, project)
    # the interpreter that submits is the one that runs: it has cg-us installed
    exe = [sys.executable, "-c",
           "import sys; from cg_us.cli import main; sys.exit(main(sys.argv[1:]))"]

    task = Task.create(
        project_name=project_name,
        task_name=name,
        script=str(LAUNCHER),
        working_directory=".",
        packages=["clearml"],
        force_single_script_file=True,
    )
    task.set_parameters({
        f"{PARAM_SECTION}/exe": json.dumps(exe),
        f"{PARAM_SECTION}/argv": json.dumps(argv),
        f"{PARAM_SECTION}/cwd": cwd,
        f"{PARAM_SECTION}/env": json.dumps(_capture_env()),
        f"{PARAM_SECTION}/queue": queue,
        f"{PARAM_SECTION}/name": name,
        f"{PARAM_SECTION}/project": project_name,
    })
    systems = _systems(argv)
    if systems:
        task.set_comment("systems: " + ", ".join(systems))
    task.add_tags(["cg-us", argv[0] if argv else "cmd"])
    Task.enqueue(task, queue_name=queue)
    info = {"id": task.id, "name": name, "queue": queue, "cwd": cwd, "project": project_name}
    try:
        info["url"] = task.get_output_log_web_page()
    except Exception:
        info["url"] = None
    return info


def queue_status(queue: str | None = None) -> list[dict]:
    """What is running and waiting on `queue`, in the order it will start."""
    _import_clearml()
    from clearml.backend_api.session.client import APIClient

    queue = queue or default_queue()
    client = APIClient()
    found = client.queues.get_all(name=f"^{queue}$", only_fields=["id", "name", "entries"])
    if not found:
        raise QueueError(f"no queue named '{queue}' on the ClearML server")
    entries = [e.task for e in (found[0].entries or [])]
    rows: list[dict] = []
    for w in client.workers.get_all():
        if queue in [q.name for q in (getattr(w, "queues", None) or [])] and getattr(w, "task", None):
            rows.append({"position": "running", "id": w.task.id, "name": w.task.name,
                         "worker": w.id})
    if entries:
        tasks = {t.id: t for t in client.tasks.get_all(id=entries, only_fields=["id", "name", "status"])}
        for pos, tid in enumerate(entries, start=1):
            t = tasks.get(tid)
            rows.append({"position": pos, "id": tid, "name": getattr(t, "name", "?"),
                         "worker": None})
    return rows


def print_submitted(info: dict) -> None:
    print(f"[queue] {info['name']}")
    print(f"        task {info['id']} enqueued on '{info['queue']}' in project {info['project']} "
          f"(cwd {info['cwd']})")
    if info.get("url"):
        print(f"        {info['url']}")
    print("        follow the order with `cg-us queue status`")
