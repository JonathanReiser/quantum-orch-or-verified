"""
snapshot_api.py — minimal transport for the Snapshot GraphQL hub.

Prefers urllib; falls back to curl because the macOS python.org framework
build ships without a usable CA bundle until `Install Certificates.command`
is run, and we do not want dataset collection to depend on that.
"""

import json
import subprocess
import urllib.request

ENDPOINT = "https://hub.snapshot.org/graphql"


def gql(query, variables=None, timeout=60):
    """Run a GraphQL query against the Snapshot hub and return its `data`."""
    payload = json.dumps({"query": query, "variables": variables or {}})
    try:
        req = urllib.request.Request(
            ENDPOINT,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
        )
        out = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception:
        proc = subprocess.run(
            ["curl", "-s", "-m", str(timeout), "-X", "POST", ENDPOINT,
             "-H", "Content-Type: application/json", "-d", payload],
            capture_output=True, text=True, check=True,
        )
        out = json.loads(proc.stdout)
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]
