"""Try to break out of a running server's code sandbox, the way a hostile visitor would.

    python scripts/check_sandbox.py http://localhost:8000

Uploads a small table, then runs code cells that try to reach the network, read the server's
secrets and files, start a program, use too much memory and run forever. Every attempt must fail;
ordinary code must still work. Exits 1 if anything got through. Standard library only.
"""

import json
import sys
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
HEADERS = {"X-Visitor-Id": f"sandbox-check-{uuid.uuid4().hex[:16]}"}

ATTEMPTS = {
    "network": "import socket\nsocket.create_connection(('1.1.1.1', 443), timeout=5)\nresult = 'GOT THROUGH'",
    "start a program": "import subprocess\nresult = 'GOT THROUGH ' + subprocess.run(['id'], capture_output=True, text=True).stdout",
    "os.system": "import os\nresult = 'GOT THROUGH' if os.system('true') == 0 else 'blocked'",
    "the app's secrets (environment)": "import os\nresult = 'GOT THROUGH' if any(k.startswith(('SUPABASE', 'GROQ', 'DATABASE', 'GOOGLE')) for k in os.environ) else 'blocked'",
    "the app's secrets (its process)": "import os\nresult = 'GOT THROUGH ' + open('/proc/1/environ').read()[:20]",
    "the app's process memory": "import os, glob\nhits = [p for p in glob.glob('/proc/[0-9]*/environ') if os.access(p, os.R_OK) and 'SUPABASE' in open(p, errors='ignore').read()]\nresult = 'GOT THROUGH' if hits else 'blocked'",
    "the app's data folder": "import os\nresult = 'GOT THROUGH ' + str(os.listdir('/app/data'))",
    "too much memory": "x = bytearray(8 * 1024 ** 3)\nresult = 'GOT THROUGH'",
    "running forever": "while True:\n    pass",
}


def call(method, path, body=None, files=None):
    headers = dict(HEADERS)
    data = None
    if files:
        boundary = uuid.uuid4().hex
        name, content = files
        data = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: text/csv\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def cell(sid, code):
    return call("POST", "/api/action", {"session_id": sid, "tool": "custom_view", "arguments": {"code": code}})


def main():
    config = call("GET", "/api/config")
    print(f"code cells: {config.get('custom_code')}, sandbox: {config.get('sandbox')}")
    if not config.get("custom_code"):
        print(f"::error title=Code sandbox::code cells are off: {config.get('code_off_reason')}")
        return 1
    csv = "x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(40))
    sid = call("POST", "/api/upload", files=("check.csv", csv.encode()))["session_id"]
    call("POST", "/api/action", {"session_id": sid, "tool": "select_target", "arguments": {"column": "y"}})

    failed = False
    ordinary = cell(sid, "result = int(train['x'].sum())")
    ok = ordinary.get("success") and ordinary.get("evidence", {}).get("output", "").isdigit()
    print(f"{'ok  ' if ok else 'FAIL'} ordinary code works: {ordinary.get('evidence') or ordinary.get('message')}")
    failed |= not ok
    for what, code in ATTEMPTS.items():
        result = cell(sid, code)
        shown = json.dumps(result.get("evidence") or result.get("message"))[:160]
        escaped = "GOT THROUGH" in shown
        print(f"{'FAIL' if escaped else 'ok  '} {what}: {shown}")
        if escaped:  # (on GitHub, a line like this becomes a visible annotation)
            print(f"::error title=Sandbox escape::{what}: {shown}")
        failed |= escaped
    after = cell(sid, "result = 2 + 2")
    alive = after.get("evidence", {}).get("output") == "4"
    print(f"{'ok  ' if alive else 'FAIL'} the sandbox still works afterwards")
    failed |= not alive
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
