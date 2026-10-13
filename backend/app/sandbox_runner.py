"""The sandbox runner: runs user code (a code cell) in its own process and reports back plain data.

Standalone on purpose (no `app` imports): it's the only program that ever executes user code, and it
also runs inside a Modal sandbox, where the rest of the app isn't. Nothing it sends back is code:
tables travel as Parquet, values as JSON, trained models as skops files (loadable without running code).

Jobs (JSON):
  view    code + train         -> what to show: `result`, print() output, matplotlib charts
  learn   code + train         -> learn(train): the plain values a Change cell learns
  apply   code + learned + df  -> apply(df, learned): the changed rows
  models  code + practice split -> each model's practice scores and the trained model (skops)
  check   code (+ train/X_learn) -> runs it (models: only builds them), reports the first error
  probe   (hardening self-test) -> which protections actually hold

How it runs:
  python sandbox_runner.py --job IN --out OUT     one job (Modal)
  python sandbox_runner.py --serve                jobs over stdin/stdout (the app's local worker)
  python sandbox_runner.py --zygote SOCKET   Linux, started as root: each job in a forked child that
      drops to a throwaway user id of its own, gets memory/CPU/file limits, and loses the network and
      the ability to start programs (seccomp), before any user code runs
"""

import ast
import base64
import contextlib
import io
import json
import os
import struct
import sys
import time
import traceback

CODE_FILENAME = "<your code>"
MAX_OUTPUT_CHARS = 5000
MAX_RESULT_BYTES = 200 * 1024 * 1024


# ── Plain data in and out ──────────────────────────────────────────────────────

def _parquet(df):
    """A table as Parquet: data only. Mixed-type text columns (which Parquet can't hold) become text."""
    import pandas as pd
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer, index=True)
    except Exception:
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: v if v is None or (isinstance(v, float) and v != v) else str(v))
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=True)
    return base64.b64encode(buffer.getvalue()).decode()


def encode(message: dict, exact_frames: bool = False) -> bytes:
    """Message to bytes. Tables travel as Parquet; exact_frames (the app sending a job, never the
    sandbox answering) uses pickle instead, to keep every dtype exactly as the app has it."""
    import pandas as pd

    def plain(v):
        if isinstance(v, pd.Series):
            v = v.to_frame()
        if isinstance(v, pd.DataFrame):
            if exact_frames:
                import pickle
                return {"__pickle__": base64.b64encode(pickle.dumps(v)).decode()}
            return {"__frame__": _parquet(v)}
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, list):
            return [plain(x) for x in v]
        return v
    return json.dumps(plain(message)).encode()


def decode(data: bytes, trust_pickle: bool = False) -> dict:
    """Bytes to message. Pickle can carry code, so it's only accepted by the sandbox reading a job the
    app sent (trust_pickle=True). The app reading an answer accepts JSON and Parquet only."""
    import pandas as pd

    def back(v):
        if isinstance(v, dict):
            if set(v) == {"__frame__"}:
                return pd.read_parquet(io.BytesIO(base64.b64decode(v["__frame__"])))
            if set(v) == {"__pickle__"}:
                if not trust_pickle:
                    raise ValueError("refused: pickled data in an answer from the sandbox")
                import pickle
                return pickle.loads(base64.b64decode(v["__pickle__"]))
            return {k: back(x) for k, x in v.items()}
        if isinstance(v, list):
            return [back(x) for x in v]
        return v
    return back(json.loads(data))


def _plain(value):
    """Learned values as JSON (the same rules as the app's recipe engine)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class UserError(Exception):
    """A message for the person whose code it is."""


def _user_error(e: Exception) -> UserError:
    lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == CODE_FILENAME]
    where = f" (line {lines[-1]})" if lines else ""
    return UserError(f"Your code raised {type(e).__name__}: {e}{where}")


# ── Running the code ──────────────────────────────────────────────────────────

def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _execute(code: str, variables: dict):
    """Run code with these variables; returns (namespace, printed output, chart images)."""
    import numpy as np
    import pandas as pd
    plt = _pyplot()
    if plt:
        plt.close("all")
    namespace = {"pd": pd, "np": np, "__name__": "__main__", **variables}
    printed = io.StringIO()
    try:
        compiled = compile(code, CODE_FILENAME, "exec")
    except SyntaxError as e:
        raise UserError(f"Your code has a syntax error on line {e.lineno}: {e.msg}") from None
    try:
        with contextlib.redirect_stdout(printed):
            exec(compiled, namespace)
    except UserError:
        raise
    except Exception as e:
        raise _user_error(e) from None
    images = []
    if plt:
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
        plt.close("all")
    return namespace, printed.getvalue()[:MAX_OUTPUT_CHARS], images


def result_evidence(value) -> dict:
    """A Look cell's `result` as something the notebook can show."""
    import pandas as pd
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if isinstance(value, pd.DataFrame):
        table = value.head(50)
        if not isinstance(table.index, pd.RangeIndex):
            table = table.reset_index()
        table.columns = [str(c) for c in table.columns]
        return {"preview": json.loads(table.to_json(orient="records", date_format="iso", default_handler=str))}
    if value is None:
        return {}
    if hasattr(value, "savefig") or type(value).__name__ in ("Figure", "Axes"):
        return {}  # a chart: it's already among the images
    return {"output": str(value)[:MAX_OUTPUT_CHARS]}


def train_only(code: str) -> bool:
    """TRAIN_ONLY read from the code's text (True/False at the top level), without running it."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRAIN_ONLY" for t in node.targets):
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, bool):
                raise UserError("TRAIN_ONLY must be set to True or False.")
            return node.value.value
    return False


def _change_functions(code: str):
    namespace, _, _ = _execute(code, {})
    for fn in ("learn", "apply"):
        if not callable(namespace.get(fn)):
            raise UserError(f"Your code must define a function called {fn}().")
    return namespace


def _call(fn, *args):
    try:
        return fn(*args)
    except UserError:
        raise
    except Exception as e:
        raise _user_error(e) from None


def run_view(job):
    train = job["frames"]["train"]
    namespace, output, images = _execute(job["code"], {"train": train, "df": train})
    evidence = result_evidence(namespace.get("result"))
    if output:
        evidence["output"] = (evidence.get("output", "") + "\n" + output).strip()
    if images:
        evidence["images"] = images
    return {"evidence": evidence}


def run_learn(job):
    namespace = _change_functions(job["code"])
    learned = _call(namespace["learn"], job["frames"]["train"].copy())
    if learned is None:
        learned = {}
    if not isinstance(learned, dict):
        raise UserError(f"learn() must return a dict (it returned {type(learned).__name__}).")
    return {"learned": _plain(learned), "train_only": train_only(job["code"])}


def run_apply(job):
    import pandas as pd
    namespace = _change_functions(job["code"])
    out = _call(namespace["apply"], job["frames"]["df"], job.get("learned") or {})
    if not isinstance(out, pd.DataFrame):
        raise UserError(f"apply() must return the rows as a DataFrame (it returned {type(out).__name__}).")
    return {"frame": out}


def _models(code, frames):
    namespace, output, _ = _execute(code, {"X_learn": frames["X_learn"].copy(), "y_learn": frames["y_learn"].iloc[:, 0].copy()})
    models = namespace.get("models")
    if not isinstance(models, dict) or not models:
        raise UserError('Your code must define `models` as a dict like {"name": SomeModel()}.')
    for name, model in models.items():
        if not (hasattr(model, "fit") and hasattr(model, "predict")):
            raise UserError(f"'{name}' isn't a model: it needs fit() and predict() methods.")
    return models, output


def run_models(job):
    """Train each model on the practice split, score it like the built-in ones, send it back as skops."""
    import numpy as np
    from sklearn.metrics import (accuracy_score, classification_report, f1_score, mean_absolute_error,
                                 mean_squared_error, r2_score)
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from skops.io import dumps

    frames = job["frames"]
    X_learn, y_learn = frames["X_learn"], frames["y_learn"].iloc[:, 0]
    X_check, y_check = frames["X_check"], frames["y_check"].iloc[:, 0]
    models, output = _models(job["code"], frames)
    cv = TimeSeriesSplit(n_splits=job["cv"]["folds"]) if job["cv"]["time"] else job["cv"]["folds"]
    results, fitted = [], {}
    for name, model in models.items():
        name = str(name)
        try:
            model.fit(X_learn, y_learn)
            predicted = model.predict(X_check)
            if job["problem_type"] == "regression":
                result = {"model": name, "r2_score": round(float(r2_score(y_check, predicted)), 4),
                          "mae": round(float(mean_absolute_error(y_check, predicted)), 4),
                          "rmse": round(float(np.sqrt(mean_squared_error(y_check, predicted))), 4)}
            else:
                result = {"model": name, "accuracy": round(float(accuracy_score(y_check, predicted)), 4),
                          "f1_score": round(float(f1_score(y_check, predicted, average="weighted", zero_division=0)), 4),
                          "report": classification_report(y_check, predicted, zero_division=0)}
            scores = cross_val_score(model, X_learn, y_learn, cv=cv)
            result["cv_mean"], result["cv_std"] = round(float(scores.mean()), 4), round(float(scores.std()), 4)
            fitted[name] = base64.b64encode(dumps(model)).decode()
        except Exception as e:
            result = {"model": name, "error": f"{type(e).__name__}: {e}"}
        results.append(result)
    return {"results": results, "fitted": fitted, "output": output}


def run_check(job):
    """Does this code run? (Models are only built, not trained.) Returns nothing; raises the first error."""
    code, frames = job["code"], job.get("frames") or {}
    if job["cell"] == "change":  # exactly what Run does: learn, then apply what was learned
        learned = run_learn({"code": code, "frames": frames})["learned"]
        run_apply({"code": code, "learned": learned, "frames": {"df": frames["train"].copy()}})
    elif job["cell"] == "models":
        _models(code, frames)
    else:
        _execute(code, {"train": frames["train"], "df": frames["train"]})
    return {}


def run_probe(job):
    """Try what sandboxed code must not be able to do; report what was stopped."""
    import socket
    import subprocess
    stopped = {}
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
        stopped["network"] = False
    except OSError:
        stopped["network"] = True
    try:
        subprocess.run(["/bin/true"], check=False, timeout=5)
        stopped["programs"] = False
    except (OSError, subprocess.SubprocessError):
        stopped["programs"] = True
    try:
        open(f"/proc/{job['app_pid']}/environ", "rb").read(1)
        stopped["app_secrets"] = False
    except OSError:
        stopped["app_secrets"] = True
    try:
        os.listdir(job["data_dir"])
        stopped["app_data"] = False
    except OSError:
        stopped["app_data"] = True
    try:
        bytearray(job["memory_mb"] * 2 * 1024 * 1024)
        stopped["memory"] = False
    except MemoryError:
        stopped["memory"] = True
    stopped["other_user"] = os.getuid() != job["app_uid"]
    return {"stopped": stopped}


JOBS = {"view": run_view, "learn": run_learn, "apply": run_apply, "models": run_models,
        "check": run_check, "probe": run_probe}


def run_job(job: dict) -> dict:
    try:
        return {"ok": True, **JOBS[job["kind"]](job)}
    except UserError as e:
        return {"ok": False, "error": str(e)}
    except MemoryError:
        return {"ok": False, "error": "Your code used more memory than a cell may use, and was stopped."}
    except Exception as e:  # the runner itself failing on odd input: still a message, never a crash
        return {"ok": False, "error": f"The code couldn't run: {type(e).__name__}: {e}"}


# ── Hardening (Linux) ──────────────────────────────────────────────────────────

# System calls a cell never needs; with them refused, code can't reach the network, start programs,
# watch other processes or change the system
FORBIDDEN = ["execve", "execveat", "ptrace", "process_vm_readv", "process_vm_writev", "mount", "umount2",
             "pivot_root", "chroot", "unshare", "setns", "bpf", "perf_event_open", "keyctl", "add_key",
             "request_key", "kexec_load", "init_module", "finit_module", "delete_module", "personality",
             "setsid", "setpgid"]  # (leaving its process group would escape being killed with it)


# Each run as root drops to a throwaway user id of its own (no account needed): two people's cells
# running at once can't see each other's files or processes, and the process cap is per run
RUN_UID_BASE, RUN_UID_SPAN = 200_000, 50_000


def run_uid() -> int:
    return RUN_UID_BASE + (os.getpid() % RUN_UID_SPAN)


def harden(workdir: str, memory_mb: int, seconds: int) -> None:
    """In a freshly forked child, before any user code: its own user, limits, no network, no programs."""
    import errno
    import resource
    import socket
    os.chdir(workdir)
    if os.getuid() == 0:
        uid = run_uid()
        os.chown(workdir, uid, uid)  # its private folder, before giving up root
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
    limit = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (seconds + 1, seconds + 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))  # no fork bombs
    import pyseccomp as seccomp
    rules = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    for family in (socket.AF_INET, socket.AF_INET6, socket.AF_NETLINK, socket.AF_PACKET):
        rules.add_rule(seccomp.ERRNO(errno.EACCES), "socket", seccomp.Arg(0, seccomp.EQ, family))
    for call in FORBIDDEN:
        try:
            rules.add_rule(seccomp.ERRNO(errno.EPERM), call)
        except Exception:
            pass  # not a system call on this architecture
    rules.load()  # sets no_new_privs too: nothing can regain privileges afterwards


def _child_run(job_bytes: bytes, write_fd: int, workdir: str, harden_it: bool) -> None:
    """In a forked child: harden, run the job, write the result, exit (never returns)."""
    try:
        limits = (json.loads(job_bytes).get("limits") or {})
        if harden_it:  # before the job's data is even unpacked
            harden(workdir, int(limits.get("memory_mb", 2048)), int(limits.get("seconds", 60)))
        else:
            os.chdir(workdir)
        job = decode(job_bytes, trust_pickle=True)  # a job from the app
        result = encode(run_job(job))
    except BaseException as e:
        result = json.dumps({"ok": False, "error": f"The sandbox couldn't run the code: {type(e).__name__}: {e}"}).encode()
    with os.fdopen(write_fd, "wb") as out:
        out.write(result)
    os._exit(0)


def run_forked(job_bytes: bytes, harden_it=False) -> bytes:
    """Run one job in a forked child with a wall-clock limit; the child is killed when time is up.
    It works in a private folder of its own, deleted afterwards."""
    import shutil
    import signal
    import tempfile
    seconds = int((json.loads(job_bytes).get("limits") or {}).get("seconds", 60))
    workdir = tempfile.mkdtemp(prefix="cell-")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        os.setpgid(0, 0)  # its own process group (killed as a whole); setpgid/setsid are then forbidden
        _child_run(job_bytes, write_fd, workdir, harden_it)
    os.close(write_fd)
    chunks, deadline = [], time.monotonic() + seconds
    os.set_blocking(read_fd, False)
    import select
    timed_out = False
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            timed_out = True
            break
        ready, _, _ = select.select([read_fd], [], [], min(left, 0.5))
        if ready:
            chunk = os.read(read_fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(map(len, chunks)) > MAX_RESULT_BYTES:
                timed_out = None
                break
    os.close(read_fd)
    # The child leads its own process group: killing the group also ends anything its code started
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)
    shutil.rmtree(workdir, ignore_errors=True)
    if timed_out:
        return json.dumps({"ok": False, "error": f"Your code ran longer than {seconds} seconds and was stopped."}).encode()
    if timed_out is None:
        return json.dumps({"ok": False, "error": "Your code produced too much output to send back."}).encode()
    data = b"".join(chunks)
    return data or json.dumps({"ok": False, "error": "Your code stopped the sandbox (it may have used too much memory)."}).encode()


# ── Ways to be started ─────────────────────────────────────────────────────────

def _read_message(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    (size,) = struct.unpack(">I", header)
    return stream.read(size)


def _write_message(stream, data: bytes) -> None:
    stream.write(struct.pack(">I", len(data)) + data)
    stream.flush()


def serve_stdio() -> None:
    """The app's local worker: jobs one after another over stdin/stdout (forked per job where possible)."""
    _warm_up()
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # stray prints must never corrupt the channel
    while True:
        job_bytes = _read_message(source)
        if job_bytes is None:
            return
        if hasattr(os, "fork"):
            result = run_forked(job_bytes)
        else:
            result = encode(run_job(decode(job_bytes, trust_pickle=True)))  # Windows: no fork; the app restarts us on a timeout
        _write_message(sink, result)


def serve_zygote(path: str) -> None:
    """Root server on a Unix socket (Linux): each job in a child that's hardened before user code runs."""
    import signal
    import socket
    _warm_up()
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # handlers reap themselves
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    group = os.environ.get("SANDBOX_CLIENT_GROUP")
    if group:
        import grp
        os.chown(path, 0, grp.getgrnam(group).gr_gid)
    os.chmod(path, 0o660)  # only the app's group can ask for jobs
    listener.listen(64)
    while True:
        conn, _ = listener.accept()
        if os.fork() == 0:  # a handler per connection: the listener never waits on one job
            listener.close()  # the job's child must never be able to accept other people's jobs
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            with conn, conn.makefile("rwb") as stream:
                job_bytes = _read_message(stream)
                if job_bytes:
                    _write_message(stream, run_forked(job_bytes, harden_it=True))
            os._exit(0)
        conn.close()


def _warm_up() -> None:
    """Load the heavy libraries once, so each forked job starts in milliseconds."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy, pandas, sklearn.ensemble, sklearn.linear_model, sklearn.model_selection  # noqa: F401,E401
    plt = _pyplot()
    if plt:
        plt.figure()
        plt.close("all")
    with contextlib.suppress(ImportError):
        import seaborn  # noqa: F401
        import skops.io  # noqa: F401


def main(argv) -> None:
    if argv[:1] == ["--serve"]:
        serve_stdio()
    elif argv[:1] == ["--zygote"]:
        serve_zygote(argv[1])
    elif argv[:1] == ["--job"]:
        job = decode(open(argv[1], "rb").read(), trust_pickle=True)
        result = encode(run_job(job))
        open(argv[argv.index("--out") + 1], "wb").write(result)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
