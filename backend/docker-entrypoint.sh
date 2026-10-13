#!/bin/sh
# Starts the code sandbox server (when CODE_SANDBOX=hardened), then the app.
#
# The sandbox server starts as root because it must switch each code run to a throwaway,
# unprivileged user id of its own. It gets an EMPTY environment (env -i): none of the app's secrets. It never
# runs user code itself; each run happens in a forked child that drops to its own user id, gets limits,
# and loses the network and the ability to start programs before any user code runs.
# The app itself runs as 'appuser', as before.
set -e

if [ "$CODE_SANDBOX" = "hardened" ]; then
    mkdir -p /run/rhombus
    chmod 755 /run/rhombus
    (
        while true; do
            env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 MPLBACKEND=Agg \
                MPLCONFIGDIR=/tmp/matplotlib-cells SANDBOX_CLIENT_GROUP=appuser \
                python -I /app/app/sandbox_runner.py --zygote /run/rhombus/sandbox.sock \
                || true
            echo "Code sandbox server stopped; restarting in 1 s" >&2
            sleep 1
        done
    ) &
fi

# The app, as the ordinary user (no root, no extra capabilities)
exec setpriv --reuid=appuser --regid=appuser --init-groups --inh-caps=-all \
    uvicorn app:app --host 0.0.0.0 --port "${PORT}" --workers 1 --timeout-keep-alive 75
