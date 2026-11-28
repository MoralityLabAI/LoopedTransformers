"""Sample once from a saved Tinker sampler checkpoint on the free model (Linux venv; standalone).

    python adapter_resample.py --sampler-path tinker://.../sampler_weights/<name> --out receipt.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_qual import EFFORT, MODEL, final_int, message_text, pointer_root  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampler-path", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    import tinker
    from tinker import types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    r = renderers.get_renderer(model_info.get_recommended_renderer_name(MODEL), get_tokenizer(MODEL), model_name=MODEL)
    svc = tinker.ServiceClient()
    sc = svc.create_sampling_client(model_path=a.sampler_path)
    sc = sc.result() if hasattr(sc, "result") and not hasattr(sc, "sample") else sc
    root = pointer_root(729001)
    prompt = r.build_generation_prompt([{"role": "user", "content": root["prompt"]}], effort=EFFORT["low"])
    t0 = time.monotonic()
    res = sc.sample(prompt=prompt, num_samples=1, sampling_params=types.SamplingParams(
        max_tokens=a.max_tokens, temperature=0.0, seed=0, stop=r.get_stop_sequences())).result()
    seq = res.sequences[0]
    text = message_text(r.parse_response(list(seq.tokens))[0])
    out = {"sampler_path": a.sampler_path, "base_model": sc.get_base_model(), "free_base": sc.get_base_model() == MODEL,
           "completion_tokens": len(seq.tokens), "stop_reason": str(getattr(seq, "stop_reason", "")),
           "answer": final_int(text), "expected": root["answer"], "correct": final_int(text) == root["answer"],
           "final_text_tail": text[-160:], "seconds": round(time.monotonic() - t0, 3),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
