"""Native Tinker SDK qualification for the free Inkling-2-Mini model (Stage A fallback).

Used after the OpenAI-compatible route returned HTTP 403. Steps, each recorded, none retried:
  1. server capabilities: is the free model id listed?
  2. sampling client on the free model id (paid ids are refused before any request)
  3. two public pointer roots, three samples each, 512 max tokens, temperature 1.0, seeded
Writes receipts without the credential, raw error bodies or reasoning text.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import resources as res
from .tinker_backend import MODEL, DEFAULT_MAX_TOKENS, guard_context, guard_model


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    res.set_below_normal_priority()
    from research_gym.ink_fp_audit_01.core import grade, roots
    receipt = {"model": guard_model(MODEL), "route": "native_sdk", "steps": [], "paid_fallback": False,
               "automatic_retries": 0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def step(name, fn):
        t0 = time.monotonic()
        try:
            out = fn()
            receipt["steps"].append({"step": name, "ok": True, "seconds": time.monotonic() - t0, **(out or {})})
            write(a.out / "receipt.json", receipt)
            return out
        except Exception as exc:
            import os
            msg = str(exc)[:240]
            key = os.environ.get("TINKER_API_KEY")
            if key:
                msg = msg.replace(key, "[redacted]")
            receipt["steps"].append({"step": name, "ok": False, "seconds": time.monotonic() - t0,
                                     "error_type": type(exc).__name__, "error_message": msg,
                                     "http_status": getattr(exc, "status_code", getattr(exc, "code", None))})
            receipt["peak_rss_bytes"] = res.process_tree_rss()
            write(a.out / "receipt.json", receipt)
            raise SystemExit(f"stopped at {name}: {type(exc).__name__}")

    import tinker
    from tinker import types
    svc = step("service_client", lambda: {"sdk_version": tinker.__version__}) and tinker.ServiceClient()

    def caps():
        c = svc.get_server_capabilities()
        names = [getattr(m, "model_name", str(m)) for m in getattr(c, "supported_models", [])]
        return {"n_models": len(names), "free_model_listed": MODEL in names,
                "inkling_ids": sorted(n for n in names if "Inkling" in n)}
    info = step("capabilities", caps)
    if not info["free_model_listed"]:
        step("free_model_check", lambda: (_ for _ in ()).throw(RuntimeError("free model id not listed")))

    holder = {}

    def open_sampler():
        s = svc.create_sampling_client(base_model=MODEL)
        holder["s"] = s.result() if hasattr(s, "result") else s  # SDK 0.40 returns a future
        holder["tok"] = holder["s"].get_tokenizer()
        return {"base_model": holder["s"].get_base_model()}
    step("sampling_client", open_sampler)

    rows, usage = [], {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    for r in roots(n=2):
        def sample(r=r):
            tok = holder["tok"]
            ids = list(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                               add_generation_prompt=True, tokenize=True))
            guard_context(len(ids), DEFAULT_MAX_TOKENS)
            out = holder["s"].sample(prompt=types.ModelInput.from_ints(ids), num_samples=a.samples,
                                     sampling_params=types.SamplingParams(max_tokens=DEFAULT_MAX_TOKENS,
                                                                          temperature=1.0, seed=729)).result()
            usage["requests"] += 1
            usage["prompt_tokens"] += len(ids) * a.samples
            graded = []
            for seq in out.sequences:
                toks = list(seq.tokens)
                usage["completion_tokens"] += len(toks)
                text = tok.decode(toks, skip_special_tokens=True)
                g = grade(text, r["answer"])
                rec = {"root_id": r["root_id"], "answer": r["answer"], "completion_tokens": len(toks),
                       "stop_reason": str(getattr(seq, "stop_reason", "")), "has_logprobs": bool(seq.logprobs),
                       "final_text_tail": text[-300:], **g}
                rows.append(rec)
                graded.append(g["correct"])
            return {"root_id": r["root_id"], "prompt_tokens": len(ids), "correct": sum(graded), "n": len(graded)}
        step(f"sample_{r['root_id']}", sample)
    with (a.out / "records.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    receipt.update({"usage": usage, "n_samples": len(rows), "correct": sum(x["correct"] for x in rows),
                    "completed": sum(x["completed"] for x in rows), "repeated": sum(x["repetition"] for x in rows),
                    "peak_rss_bytes": res.process_tree_rss(), "verdict": "development_only"})
    write(a.out / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("usage", "n_samples", "correct", "completed", "repeated", "peak_rss_bytes")}))


if __name__ == "__main__":
    main()
