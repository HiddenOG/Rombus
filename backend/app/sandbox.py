"""Where user code runs: never inside the app's own process.

Every code cell goes to app/sandbox_runner.py in another process, and only plain data comes back.
Which process is set by CODE_SANDBOX:

  local     (your own machine) a warm worker process: a time limit and a fresh namespace per job,
            no other protection, because it's your code on your computer
  hardened  (the server) a root "zygote" started next to the app (see the Dockerfile) forks a child per
            job that runs as another user, with memory/CPU limits, no network and no starting programs.
            Checked by a self-test before first use; if the protections don't all hold (or the zygote
            isn't running), the Modal fallback is used if configured, otherwise code stays off
  modal     every job in a Modal sandbox (gVisor isolation, network blocked), using MODAL_TOKEN_ID/SECRET
"""

import json
import logging
import os
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from app.config import settings
from app import sandbox_runner as runner

logger = logging.getLogger(__name__)

RUNNER = Path(runner.__file__)


class SandboxUnavailable(Exception):
    pass


def run(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one job wherever code runs on this server. Returns the runner's result (frames decoded);
    raises ValueError with a message for the user when the code failed or can't run here."""
    if not settings.ALLOW_CUSTOM_CODE:
        raise ValueError("Running code is switched off on this server.")
    seconds = settings.CODE_TIME_LIMIT_SECONDS * (5 if job["kind"] == "models" else 1)
    job = {**job, "limits": {"seconds": seconds, "memory_mb": settings.CODE_MEMORY_MB}}
    try:
        backend = _backend()
    except SandboxUnavailable:
        raise ValueError("Code can't run right now: the code sandbox isn't available. The built-in steps "
                         "work as usual; try code again in a minute.") from None
    result = backend.run(job)
    if not result.get("ok"):
        raise ValueError(result.get("error") or "The code failed.")
    return result


CODE_TOOLS = {"custom_view", "custom_code", "custom_models"}


def not_allowed(session) -> Optional[str]:
    """Why this notebook may not run code (None when it may)."""
    if not settings.ALLOW_CUSTOM_CODE:
        return "Running code is switched off on this server."
    if settings.CODE_FOR == "accounts" and not (session.owner or "").startswith("user:"):
        return ("Running code needs an account: sign in (your notebook comes with you). Until then, the "
                "built-in steps and the assistant work as usual, and Export gives you the code to run at home.")
    return None


def status() -> Dict[str, Any]:
    """What /api/health says about running code."""
    if not settings.ALLOW_CUSTOM_CODE:
        return {"enabled": False}
    try:
        backend = _backend()
        return {"enabled": True, "sandbox": backend.name, "protections": getattr(backend, "protections", None)}
    except SandboxUnavailable as e:
        return {"enabled": False, "reason": str(e)}


# ── Choosing the backend (once) ────────────────────────────────────────────────

_chosen = None
_choosing = threading.Lock()
_unavailable_until = 0.0
RECHECK_SECONDS = 30  # the sandbox server may still be starting: look again after this long


def _backend():
    """The backend in use. A failure isn't remembered forever (the sandbox server starts alongside
    the app, and may just not be up yet): it's checked again after RECHECK_SECONDS."""
    import time
    global _chosen, _unavailable_until
    with _choosing:
        if _chosen is None:
            if time.monotonic() < _unavailable_until:
                raise _last_reason
            try:
                _chosen = _choose()
            except SandboxUnavailable as e:
                _remember_failure(e)
                _unavailable_until = time.monotonic() + RECHECK_SECONDS
                raise
        return _chosen


_last_reason = SandboxUnavailable("not checked yet")


def _remember_failure(e: SandboxUnavailable) -> None:
    global _last_reason
    _last_reason = e


def _choose():
    wanted = settings.CODE_SANDBOX
    if wanted == "local":
        return LocalWorker()
    if wanted == "modal":
        return ModalSandbox()
    if wanted == "hardened":
        try:
            return Zygote.checked()
        except SandboxUnavailable as e:
            if settings.MODAL_TOKEN_ID and settings.MODAL_TOKEN_SECRET:
                logger.warning("Hardened sandbox unavailable (%s): using Modal instead", e)
                return ModalSandbox()
            logger.error("Hardened sandbox unavailable (%s) and no Modal fallback: code stays off", e)
            raise
    raise SandboxUnavailable(f"Unknown CODE_SANDBOX '{wanted}' (use local, hardened or modal).")


def reset() -> None:
    """Forget the chosen backend (tests, or after changing settings)."""
    global _chosen, _unavailable_until
    with _choosing:
        if isinstance(_chosen, LocalWorker):
            _chosen.stop()
        _chosen = None
        _unavailable_until = 0.0


def _frames(message: bytes) -> Dict[str, Any]:
    if len(message) > runner.MAX_RESULT_BYTES:
        return {"ok": False, "error": "Your code produced too much output to send back."}
    try:
        return runner.decode(message)  # JSON and Parquet only: nothing in it can run
    except Exception:
        return {"ok": False, "error": "The sandbox sent back something unreadable."}


# ── local: a warm worker process ───────────────────────────────────────────────

class LocalWorker:
    """One long-lived runner process (libraries loaded once). Jobs go one at a time; a job that runs
    past its time limit gets the worker killed and restarted."""
    name = "local"

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _start(self):
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE",
                                                           "LANG", "MPLCONFIGDIR", "MPLBACKEND")}
        self._process = subprocess.Popen([sys.executable, "-I", str(RUNNER), "--serve"], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, env=env, cwd=str(RUNNER.parent))

    def stop(self):
        if self._process and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._process = None

    def run(self, job):
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()
            data = runner.encode(job, exact_frames=True)
            reply: Dict[str, Any] = {}

            def talk():
                try:
                    runner._write_message(self._process.stdin, data)
                    message = runner._read_message(self._process.stdout)
                    reply["message"] = message
                except (OSError, ValueError):
                    pass
            # The worker enforces the limit itself where it can fork; this is the backstop (Windows)
            talker = threading.Thread(target=talk, daemon=True)
            talker.start()
            talker.join(job["limits"]["seconds"] + 5)
            if talker.is_alive() or reply.get("message") is None:
                self.stop()
                if talker.is_alive():
                    return {"ok": False, "error": f"Your code ran longer than {job['limits']['seconds']} seconds and was stopped."}
                return {"ok": False, "error": "Your code stopped the worker (it may have used too much memory)."}
            return _frames(reply["message"])


# ── hardened: the zygote next to the app ───────────────────────────────────────

class Zygote:
    name = "hardened"

    def __init__(self, path: str):
        self.path = path
        self.protections: Dict[str, bool] = {}

    @classmethod
    def checked(cls) -> "Zygote":
        path = settings.SANDBOX_SOCKET
        if not hasattr(socket, "AF_UNIX") or not os.path.exists(path):
            raise SandboxUnavailable(f"the sandbox server isn't running (no socket at {path})")
        zygote = cls(path)
        probe = zygote.run({"kind": "probe", "app_pid": os.getpid(), "app_uid": os.getuid(),
                            "data_dir": str(Path(settings.DATA_DIR).resolve()),
                            "limits": {"seconds": 30, "memory_mb": 256}, "memory_mb": 256})
        stopped = probe.get("stopped") or {}
        zygote.protections = stopped
        missing = [name for name in ("network", "programs", "app_secrets", "app_data", "memory", "other_user")
                   if not stopped.get(name)]
        if not probe.get("ok") or missing:
            raise SandboxUnavailable(f"self-test failed: not protected against {', '.join(missing) or probe.get('error')}")
        logger.info("Hardened sandbox ready: %s", ", ".join(stopped))
        return zygote

    def run(self, job):
        data = runner.encode(job, exact_frames=True)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(job["limits"]["seconds"] + 30)
                conn.connect(self.path)
                conn.sendall(struct.pack(">I", len(data)) + data)
                with conn.makefile("rb") as stream:
                    message = runner._read_message(stream)
        except OSError as e:
            return {"ok": False, "error": f"The code sandbox isn't answering ({e}). Try again in a moment."}
        if message is None:
            return {"ok": False, "error": "The code sandbox closed without an answer."}
        return _frames(message)


# ── modal: a hosted sandbox ────────────────────────────────────────────────────

class ModalSandbox:
    """Each job in a Modal sandbox (gVisor, network blocked). One sandbox is kept warm and reused
    for a few minutes; each job runs as its own process in it, with the job's time limit."""
    name = "modal"
    IDLE_SECONDS = 300

    def __init__(self):
        if not (settings.MODAL_TOKEN_ID and settings.MODAL_TOKEN_SECRET):
            raise SandboxUnavailable("CODE_SANDBOX=modal needs MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
        os.environ.setdefault("MODAL_TOKEN_ID", settings.MODAL_TOKEN_ID)
        os.environ.setdefault("MODAL_TOKEN_SECRET", settings.MODAL_TOKEN_SECRET)
        self._sandbox = None
        self._lock = threading.Lock()
        self.protections = {"isolation": "gvisor", "network": True}

    def _image(self):
        import modal
        return (modal.Image.debian_slim(python_version="3.12")
                .pip_install("pandas>=3.0,<4", "numpy>=2.3", "scikit-learn>=1.9", "scipy>=1.18", "pyarrow>=25",
                             "matplotlib>=3.9", "seaborn>=0.13", "skops>=0.16", "xgboost-cpu>=2.1")
                .env({"MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1"})
                .add_local_file(str(RUNNER), "/sandbox/runner.py"))

    def _ready(self):
        import modal
        if self._sandbox is not None and self._sandbox.poll() is None:
            return self._sandbox
        app = modal.App.lookup("rhombus-sandbox", create_if_missing=True)
        self._sandbox = modal.Sandbox.create(
            "sleep", "infinity", app=app, image=self._image(), block_network=True, cpu=1.0,
            memory=settings.CODE_MEMORY_MB, timeout=3600, idle_timeout=self.IDLE_SECONDS)
        return self._sandbox

    def run(self, job):
        import uuid
        name = uuid.uuid4().hex
        with self._lock:
            try:
                box = self._ready()
                box.filesystem.write_bytes(runner.encode(job, exact_frames=True), f"/tmp/{name}.job")
                process = box.exec("python", "-I", "/sandbox/runner.py", "--job", f"/tmp/{name}.job",
                                   "--out", f"/tmp/{name}.out", timeout=job["limits"]["seconds"])
                process.wait()
                if process.returncode != 0:
                    return {"ok": False, "error": f"Your code ran longer than {job['limits']['seconds']} seconds, or "
                                                  "used too much memory, and was stopped."}
                message = box.filesystem.read_bytes(f"/tmp/{name}.out")
                box.filesystem.remove(f"/tmp/{name}.job")
                box.filesystem.remove(f"/tmp/{name}.out")
            except Exception as e:
                logger.exception("Modal sandbox failed")
                self._sandbox = None
                return {"ok": False, "error": f"The code sandbox (Modal) failed: {type(e).__name__}. Try again."}
        return _frames(message)


# ── Loading trained models safely ─────────────────────────────────────────────

def load_model(encoded: str):
    """A model trained in the sandbox, loaded WITHOUT running code: skops rebuilds only the types we
    trust (scikit-learn, numpy, builtins). Anything else is refused."""
    import base64
    from skops.io import get_untrusted_types, loads
    data = base64.b64decode(encoded)
    untrusted = get_untrusted_types(data=data)
    refused = sorted(t for t in untrusted if not _trusted_type(t))
    if refused:
        raise ValueError(f"Models must be built from scikit-learn parts; this one uses {', '.join(refused[:3])}.")
    return loads(data, trusted=untrusted)


def _trusted_type(path: str) -> bool:
    """scikit-learn and XGBoost classes, numpy/scipy data types, numpy maths functions (log1p, sqrt...) and
    scikit-learn's scoring functions (a tuned search keeps its scorer, e.g. r2_score). No other functions:
    a model mustn't be able to carry a reference to something that acts (files, processes)."""
    import importlib
    import inspect
    import numpy as np
    module, _, name = path.rpartition(".")
    if module.split(".")[0] not in ("sklearn", "numpy", "scipy", "xgboost"):
        return False
    try:
        obj = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError, ValueError):
        return False
    scoring_function = inspect.isfunction(obj) and module.startswith("sklearn.metrics")
    return isinstance(obj, (type, np.ufunc)) or scoring_function


def frame(df: pd.DataFrame) -> pd.DataFrame:
    """A copy safe to send (column names as text, as Parquet needs)."""
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return out


def describe(result: Dict[str, Any]) -> str:  # pragma: no cover - debugging aid
    return json.dumps({k: v for k, v in result.items() if k != "frame"}, default=str)[:500]
