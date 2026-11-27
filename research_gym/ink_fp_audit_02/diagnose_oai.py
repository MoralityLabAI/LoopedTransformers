"""One-request diagnosis of the OpenAI-compatible Tinker route (free model only).

Sends the audit-01 smoke payload once with a generic client User-Agent and records the HTTP
status plus a short body excerpt with the credential scrubbed. No retries, no redirects.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from research_gym.ink_fp_audit_01 import tinker_remote as r01
from research_gym.ink_fp_audit_01.core import roots

UA = "ink-fp-audit/0.2 (python)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    key = os.environ["TINKER_API_KEY"]
    payload = r01.request_payload(roots(n=1)[0]["prompt"], "low", 512)
    assert payload["model"] == r01.MODEL
    req = urllib.request.Request(r01.URL, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "User-Agent": UA, "Accept": "application/json"})
    t0 = time.monotonic()
    rec = {"user_agent": UA, "model": r01.MODEL, "endpoint": r01.URL}
    try:
        with urllib.request.build_opener(r01.NoRedirect()).open(req, timeout=120) as res:
            body = res.read(2 * 1024 * 1024 + 1).decode("utf-8", "replace")
            rec.update(status=res.status, ok=True)
            data = json.loads(body)
            rec.update(usage=data.get("usage"), response_model=data.get("model"),
                       finish_reason=data["choices"][0].get("finish_reason"),
                       final_text_tail=(data["choices"][0]["message"].get("content") or "")[-300:])
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace")
        rec.update(status=exc.code, ok=False, body_excerpt=body[:300].replace(key, "[redacted]"),
                   server=exc.headers.get("server"), content_type=exc.headers.get("content-type"))
    rec["seconds"] = time.monotonic() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: rec.get(k) for k in ("status", "ok", "server", "content_type", "body_excerpt", "usage")}))


if __name__ == "__main__":
    main()
