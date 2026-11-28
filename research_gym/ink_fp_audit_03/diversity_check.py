"""Does num_samples>1 with a fixed seed return identical samples? Two requests (Linux venv).

    python diversity_check.py --out receipt.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_qual import EFFORT, MODEL, pointer_root  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import tinker
    from tinker import types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    r = renderers.get_renderer(model_info.get_recommended_renderer_name(MODEL), get_tokenizer(MODEL), model_name=MODEL)
    sc = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
    sc = sc.result() if hasattr(sc, "result") else sc
    prompt = r.build_generation_prompt([{"role": "user", "content": pointer_root(729003)["prompt"]}], effort=EFFORT["low"])
    out = {"model": MODEL, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for label, seed in (("fixed_seed_729", 729), ("no_seed", None)):
        res = sc.sample(prompt=prompt, num_samples=4, sampling_params=types.SamplingParams(
            max_tokens=1024, temperature=1.0, seed=seed, stop=r.get_stop_sequences())).result()
        seqs = [tuple(s.tokens) for s in res.sequences]
        out[label] = {"n": len(seqs), "distinct_sequences": len(set(seqs)), "lengths": [len(s) for s in seqs]}
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
