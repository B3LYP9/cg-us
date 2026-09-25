"""ClearML task body for `cg-us <command> --enqueue`.

Uploaded with the task as a single script and run by the clearml-agent in its own
environment, so it may only import the standard library and clearml. It runs the
command recorded by `cg-us.clearml_queue.submit` - same interpreter, working
directory and PATH as the shell that queued it - and forwards the output to the
task console. The exit code becomes the task status.
"""

import json
import os
import shutil
import signal
import subprocess
import sys

from clearml import Task

task = Task.current_task() or Task.init(project_name="cg-us", task_name="cg-us job")
params = task.get_parameters_as_dict().get("cgus", {})
exe = json.loads(params["exe"])
argv = json.loads(params["argv"])
cwd = params["cwd"]

env = dict(os.environ)
env.update(json.loads(params.get("env") or "{}"))
env["PYTHONUNBUFFERED"] = "1"
if not shutil.which("gmx", path=env.get("PATH", "")) and os.path.exists("/usr/local/gromacs/bin/gmx"):
    env["PATH"] = "/usr/local/gromacs/bin" + os.pathsep + env.get("PATH", "")

print(f"[cg-us queue] cd {cwd}", flush=True)
print("[cg-us queue] cg-us " + " ".join(argv), flush=True)
proc = subprocess.Popen(exe + argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                        start_new_session=True)


def _stop(signum, _frame):
    # aborting the task must not leave mdrun behind: the whole group goes
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
task.set_parameter("cgus/exit_code", str(rc))
sys.exit(rc)
