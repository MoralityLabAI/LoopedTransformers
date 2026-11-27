"""Free-tier Tinker behavioral development pilot; never edits model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from .core import digest, grade, roots

MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions"
EFFORTS = ("low", "medium", "high")


def plan(n_roots=2, max_tokens=512, efforts=EFFORTS):
    if not 2 <= n_roots <= 8 or not 1 <= max_tokens <= 2048:
        raise ValueError("require 2..8 roots and 1..2048 completion tokens")
    if not efforts or len(set(efforts)) != len(efforts) or not set(efforts) <= set(EFFORTS):
        raise ValueError("invalid or duplicate effort")
    rows = roots(n=n_roots)
    jobs = []
    for i, root in enumerate(rows):
        order = list(efforts)
        random.Random(728000+i).shuffle(order)
        for effort in order:
            jobs.append({"root_id": root["root_id"], "effort": effort})
    manifest = {
        "schema": "ink-fp-audit-01.tinker.dev.v1", "model": MODEL, "endpoint": URL,
        "roots": rows, "jobs": jobs, "max_completion_tokens": max_tokens,
        "temperature": 0, "request_cap": len(jobs),
        "max_requested_completion_tokens": len(jobs)*max_tokens,
        "paid_fallback": False, "automatic_retries": 0, "concurrency": 1,
        "unit": "generated source mapping", "n_units": n_roots,
        "n_cases": len(jobs), "sealed_readout": False,
        "primary_contrast": "high vs low reasoning effort if both registered",
        "physical_dose": "requested reasoning_effort; internal loop count unobserved",
        "scientific_effect_floor": "exploratory 0.10 correctness gain",
        "matched_sham_floor": None, "planned_comparisons": max(0, len(efforts)-1),
        "estimated_resolution": "not estimated; public engineering pilot",
        "readout_selection_exposure": "public development",
        "causal_controls": "same root, prompt, temperature and completion cap; no activation intervention",
        "verdict": "development_only", "internal_recurrence_claim": False,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def request_payload(prompt, effort, max_tokens):
    if effort not in EFFORTS or not 1 <= max_tokens <= 2048:
        raise ValueError("unregistered effort or completion cap")
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": effort, "max_completion_tokens": max_tokens,
            "temperature": 0, "stream": False}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def sample(key, payload, timeout=120):
    if payload.get("model") != MODEL:
        raise ValueError("paid or unregistered model blocked")
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": "Bearer "+key, "Content-Type": "application/json"},
    )
    # Do not forward credentials across redirects. No automatic retries/fallback.
    with urllib.request.build_opener(NoRedirect()).open(req, timeout=timeout) as res:
        return json.loads(res.read(2*1024*1024+1))


def score_response(response, expected_answer):
    if not isinstance(response, dict) or len(response.get("choices", [])) != 1:
        raise ValueError("expected exactly one completion")
    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("expected text completion")
    result = grade(content, expected_answer)
    return {"content": content, **result, "finish_reason": choice.get("finish_reason"),
            "response_model": response.get("model"), "response_id": response.get("id"),
            "usage": response.get("usage"),
            "reasoning_characters": len(message.get("reasoning_content") or "")}


def write_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def run(manifest, out, key, transport=sample, delay=1.0, timeout=120):
    if not key or manifest["model"] != MODEL or manifest["endpoint"] != URL:
        raise ValueError("missing credential or unregistered endpoint/model")
    if delay < 0 or timeout <= 0:
        raise ValueError("invalid delay/timeout")
    # Reconstruct the plan before sending anything; never accept an expanded job list.
    expected = plan(manifest["n_units"], manifest["max_completion_tokens"],
                    tuple(dict.fromkeys(j["effort"] for j in manifest["jobs"])))
    # Job order depends on registered effort order. Compare the fixed cell set and roots.
    actual_cells = [(j["root_id"], j["effort"]) for j in manifest["jobs"]]
    expected_cells = [(j["root_id"], j["effort"]) for j in expected["jobs"]]
    if sorted(actual_cells) != sorted(expected_cells) or manifest["roots"] != expected["roots"]:
        raise ValueError("unregistered jobs")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"manifest.json", manifest)
    records = out/"records.jsonl"
    records.touch()
    lookup = {r["root_id"]: r for r in manifest["roots"]}
    results = []
    for index, job in enumerate(manifest["jobs"]):
        root = lookup[job["root_id"]]
        start = time.monotonic()
        try:
            reply = transport(key, request_payload(root["prompt"], job["effort"],
                                                   manifest["max_completion_tokens"]), timeout)
            rec = {**job, **score_response(reply, root["answer"]),
                   "answer": root["answer"], "seconds": time.monotonic()-start}
            # Keep the requested model fixed; report returned identity for review.
            with records.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, allow_nan=False)+"\n")
            results.append(rec)
        except Exception as exc:
            # No raw error strings or HTTP bodies (may contain sensitive data).
            write_json(out/"failure.json", {"job": job, "attempted_requests": index+1,
                      "completed_requests": len(results), "error_type": type(exc).__name__,
                      "http_status": getattr(exc, "code", None),
                      "automatic_retries": 0, "paid_fallback": False})
            raise RuntimeError(f"Tinker pilot stopped: {type(exc).__name__}") from None
        if index+1 < len(manifest["jobs"]) and delay:
            time.sleep(delay)

    summary = {}
    for effort in EFFORTS:
        lane = [r for r in results if r["effort"] == effort]
        if lane:
            summary[effort] = {"n": len(lane), "correct": sum(r["correct"] for r in lane),
                               "completed": sum(r["completed"] for r in lane),
                               "repeated": sum(r["repetition"] for r in lane)}
    write_json(out/"complete.json", {"arms": summary, "requests": len(results),
               "records_sha256": digest(records), "verdict": "development_only"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="explicitly enable remote requests")
    p.add_argument("--out", type=Path)
    p.add_argument("--roots", type=int, default=2)
    p.add_argument("--max-completion-tokens", type=int, default=512)
    p.add_argument("--efforts", nargs="+", choices=EFFORTS, default=list(EFFORTS))
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=120)
    a = p.parse_args()
    m = plan(a.roots, a.max_completion_tokens, tuple(a.efforts))
    if not a.run:
        print(json.dumps(m, indent=2))
        return
    key = os.environ.get("TINKER_API_KEY")
    if not key or a.out is None:
        p.error("--run needs TINKER_API_KEY in environment and a fresh --out")
    run(m, a.out, key, delay=a.delay, timeout=a.timeout)


if __name__ == "__main__":
    main()
