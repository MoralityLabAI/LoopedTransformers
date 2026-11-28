"""Native Tinker qualification for free Inkling-2-Mini (INK-FP-AUDIT-03, Stage B).

Runs on Linux (tml-renderers ships Linux/macOS wheels only), e.g. inside WSL with a venv that
has tinker, tinker-cookbook and tml-renderers. Standalone: imports nothing from this repository,
so it can run from a read-only checkout. The credential comes from TINKER_API_KEY in the
environment and is never written. Every step is recorded; none is retried.

    python native_qual.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
EFFORT = {"low": 0.2, "medium": 0.7, "high": 0.9}  # tinker-docs cookbook/inkling/thinking-effort presets
CONTEXT_CEILING = 16_384


def pointer_root(seed: int) -> dict:
    """Same generator as research_gym.ink_fp_audit_01.core.roots (copied: script is standalone)."""
    rng = random.Random(seed)
    perm = list(range(11))
    rng.shuffle(perm)
    start = rng.randrange(11)
    hops = rng.randint(3, 8)
    answer = start
    for _ in range(hops):
        answer = perm[answer]
    table = ", ".join(f"{k}->{v}" for k, v in enumerate(perm))
    prompt = (f"Follow the mapping {table}. Start at {start} and apply the mapping "
              f"exactly {hops} times. Finish with a separate line ANSWER: <integer>.")
    return {"root_id": f"pointer-{seed}", "prompt": prompt, "answer": answer}


def final_int(text: str):
    m = re.findall(r"(?m)^ANSWER:\s*([+-]?\d+)\s*$", text or "")
    return int(m[-1]) if len(m) == 1 else None


def message_text(msg) -> str:
    """Visible answer text of a parsed assistant message (content may be a list of parts)."""
    text = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", msg)
    if isinstance(text, list):
        text = "".join(p.get("text", "") if isinstance(p, dict) and p.get("type", "text") == "text"
                       else (getattr(p, "text", "") or "") for p in text)
    return str(text or "")


def rss() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    key = os.environ.get("TINKER_API_KEY")
    if not key:
        raise SystemExit("TINKER_API_KEY not present in environment")
    receipt = {"schema": "ink-fp-audit-03.native_qual.v1", "model": MODEL, "steps": [],
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "automatic_retries": 0,
               "paid_fallback": False, "effort_presets": EFFORT}
    ctx: dict = {}

    def save():
        receipt["peak_rss_bytes"] = rss()
        (a.out / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")

    def step(name, fn, needs=()):
        if any(n not in ctx for n in needs):
            receipt["steps"].append({"step": name, "ok": False, "skipped": True, "missing": [n for n in needs if n not in ctx]})
            save()
            return None
        t0 = time.monotonic()
        try:
            info = fn() or {}
            receipt["steps"].append({"step": name, "ok": True, "seconds": round(time.monotonic() - t0, 3), **info})
        except Exception as exc:
            msg = str(exc)[:300].replace(key, "[redacted]")
            receipt["steps"].append({"step": name, "ok": False, "seconds": round(time.monotonic() - t0, 3),
                                     "error_type": type(exc).__name__, "error_message": msg})
        save()

    import tinker
    from tinker import types
    import importlib.metadata as md
    receipt["versions"] = {p: md.version(p) for p in ("tinker", "tinker-cookbook", "tml-renderers", "torch")}

    def resolve(x):
        return x.result() if hasattr(x, "result") and not hasattr(x, "sample") and not hasattr(x, "forward_backward") else x

    # 1. renderer and tokenizer -----------------------------------------------------------------
    def renderer():
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        name = model_info.get_recommended_renderer_name(MODEL)
        tok = get_tokenizer(MODEL)
        ctx["r"] = renderers.get_renderer(name, tok, model_name=MODEL)
        ctx["tok"] = tok
        root = pointer_root(729001)
        ctx["root"] = root
        msgs = [{"role": "user", "content": root["prompt"]}]
        ctx["msgs"] = msgs
        lens = {e: ctx["r"].build_generation_prompt(msgs, effort=v).length for e, v in EFFORT.items()}
        ctx["prompt_lens"] = lens
        return {"renderer": name, "prompt_tokens_by_effort": lens,
                "stop_sequences": str(ctx["r"].get_stop_sequences())[:200]}
    step("renderer_tokenizer", renderer)

    # 2. service + sampling -------------------------------------------------------------------------
    def service():
        ctx["svc"] = tinker.ServiceClient()
        ctx["sampler"] = resolve(ctx["svc"].create_sampling_client(base_model=MODEL))
        return {"base_model": ctx["sampler"].get_base_model()}
    step("sampling_client", service)

    def sample_effort(effort, n=2, temperature=1.0, seed=729):
        r = ctx["r"]
        prompt = r.build_generation_prompt(ctx["msgs"], effort=EFFORT[effort])
        if prompt.length + a.max_tokens > CONTEXT_CEILING:
            raise ValueError("context ceiling")
        res = ctx["sampler"].sample(prompt=prompt, num_samples=n, sampling_params=types.SamplingParams(
            max_tokens=a.max_tokens, temperature=temperature, seed=seed, stop=r.get_stop_sequences())).result()
        out = []
        for seq in res.sequences:
            toks = list(seq.tokens)
            lps = list(seq.logprobs) if seq.logprobs is not None else None
            msg, ok = r.parse_response(toks)
            text = message_text(msg)
            out.append({"tokens": toks, "logprobs": lps, "text": text, "parse_ok": bool(ok),
                        "stop_reason": str(getattr(seq, "stop_reason", ""))})
        return prompt, out

    def sampling():
        rows = {}
        for e in EFFORT:
            prompt, out = sample_effort(e)
            ctx.setdefault("samples", {})[e] = (prompt, out)
            rows[e] = [{"completion_tokens": len(o["tokens"]), "logprobs_len": None if o["logprobs"] is None else len(o["logprobs"]),
                        "logprobs_match_tokens": o["logprobs"] is not None and len(o["logprobs"]) == len(o["tokens"]),
                        "parse_ok": o["parse_ok"], "stop_reason": o["stop_reason"],
                        "answer": final_int(o["text"]), "correct": final_int(o["text"]) == ctx["root"]["answer"],
                        "final_text_tail": o["text"][-160:]} for o in out]
        return {"by_effort": rows}
    step("sample_with_logprobs", sampling, needs=("r", "sampler"))

    def greedy_vs_oai():
        prompt, out = sample_effort("low", n=1, temperature=0.0, seed=0)
        return {"native_low_temp0_prompt_tokens": prompt.length, "oai_low_prompt_tokens_audit02": 98,
                "prompt_token_match": prompt.length == 98, "native_final_tail": out[0]["text"][-60:],
                "oai_final_tail_audit02": "ANSWER: 3", "final_answer_match": final_int(out[0]["text"]) == 3}
    step("native_vs_oai_consistency", greedy_vs_oai, needs=("r", "sampler"))

    # 3. training -------------------------------------------------------------------------------------
    def training_client():
        ctx["tc"] = resolve(ctx["svc"].create_lora_training_client(base_model=MODEL, rank=8, seed=0))
        info = ctx["tc"].get_info()
        return {"info": str(info)[:300]}
    step("lora_training_client", training_client, needs=("svc",))

    def td(values, dtype):
        return types.TensorData(data=list(values), dtype=dtype, shape=[len(values)])

    def rl_datum(prompt, toks, lps, adv, gspo=False):
        p = prompt.to_ints()
        full = p + toks
        n_p = len(p)
        inputs = {"target_tokens": td(full[1:], "int64"),
                  "logprobs": td([0.0] * (n_p - 1) + [float(x) for x in lps], "float32"),
                  "advantages": td([0.0] * (n_p - 1) + [float(adv)] * len(toks), "float32")}
        if gspo:
            inputs["sequence_ratio_mask"] = td([0.0] * (n_p - 1) + [1.0] * len(toks), "float32")
        return types.Datum(model_input=types.ModelInput.from_ints(full[:-1]), loss_fn_inputs=inputs)

    def ce():
        root = ctx["root"]
        msgs = ctx["msgs"] + [{"role": "assistant", "content": f"ANSWER: {root['answer']}"}]
        mi, w = ctx["r"].build_supervised_example(msgs, effort=EFFORT["low"])
        toks = mi.to_ints()
        w = [float(x) for x in w.tolist()]
        d = types.Datum(model_input=types.ModelInput.from_ints(toks[:-1]),
                        loss_fn_inputs={"target_tokens": td(toks[1:], "int64"), "weights": td(w[1:], "float32")})
        fb = ctx["tc"].forward_backward([d], "cross_entropy").result()
        ctx["tc"].optim_step(types.AdamParams(learning_rate=1e-5)).result()
        return {"tokens": len(toks), "trained_tokens": int(sum(w)), "metrics": dict(getattr(fb, "metrics", {}) or {})}
    step("cross_entropy_step", ce, needs=("tc", "r"))

    def rl(loss):
        prompt, out = ctx["samples"]["low"]
        usable = [o for o in out if o["logprobs"] is not None and len(o["logprobs"]) == len(o["tokens"]) and o["tokens"]]
        if len(usable) < 1:
            raise ValueError("no sample with token-aligned logprobs")
        rewards = [1.0 if final_int(o["text"]) == ctx["root"]["answer"] else 0.0 for o in usable]
        mu = sum(rewards) / len(rewards)
        advs = [r - mu for r in rewards] if len(set(rewards)) > 1 else [1.0] * len(rewards)
        data = [rl_datum(prompt, o["tokens"], o["logprobs"], adv, gspo=(loss == "gspo")) for o, adv in zip(usable, advs)]
        cfg = {"clip_low_threshold": 1 - 3e-4, "clip_high_threshold": 1 + 4e-4} if loss == "gspo" else None
        fb = ctx["tc"].forward_backward(data, loss, cfg).result()
        ctx["tc"].optim_step(types.AdamParams(learning_rate=1e-5)).result()
        return {"datums": len(data), "advantages": advs, "behavior_policy": "native sampler, effort 0.2, temperature 1.0",
                "metrics": dict(getattr(fb, "metrics", {}) or {})}
    step("importance_sampling_step", lambda: rl("importance_sampling"), needs=("tc", "samples"))
    step("gspo_step", lambda: rl("gspo"), needs=("tc", "samples"))

    def persist():
        st = ctx["tc"].save_state("ink03-qual").result()
        sw = ctx["tc"].save_weights_for_sampler("ink03-qual-sampler").result()
        ctx["state_path"], ctx["sampler_path"] = st.path, sw.path
        return {"state_path": st.path, "sampler_path": sw.path}
    step("save_checkpoint", persist, needs=("tc",))

    def sample_trained():
        sc = resolve(ctx["svc"].create_sampling_client(model_path=ctx["sampler_path"]))
        base = sc.get_base_model()
        prompt = ctx["r"].build_generation_prompt(ctx["msgs"], effort=EFFORT["low"])
        res = sc.sample(prompt=prompt, num_samples=1, sampling_params=types.SamplingParams(
            max_tokens=a.max_tokens, temperature=0.0, seed=0, stop=ctx["r"].get_stop_sequences())).result()
        text = message_text(ctx["r"].parse_response(list(res.sequences[0].tokens))[0])
        return {"base_model": base, "free_base": base == MODEL, "answer": final_int(text),
                "correct": final_int(text) == ctx["root"]["answer"], "final_text_tail": text[-120:]}
    step("sample_from_adapter", sample_trained, needs=("sampler_path", "r"))

    def restore():
        tc2 = resolve(ctx["svc"].create_training_client_from_state_with_optimizer(ctx["state_path"]))
        prompt, out = ctx["samples"]["low"]
        o = out[0]
        fb = tc2.forward_backward([rl_datum(prompt, o["tokens"], o["logprobs"], 1.0)], "importance_sampling").result()
        tc2.optim_step(types.AdamParams(learning_rate=1e-5)).result()
        return {"restored_from": ctx["state_path"], "post_restore_step_ok": True,
                "metrics": dict(getattr(fb, "metrics", {}) or {})}
    step("restore_with_optimizer", restore, needs=("state_path", "samples"))

    def billing():
        rest = ctx["svc"].create_rest_client()
        names = [m for m in dir(rest) if not m.startswith("_") and any(k in m for k in ("bill", "usage", "credit", "balance"))]
        out = {"rest_methods_matching": [n for n in names if not n.endswith("_async")]}
        # Only numeric fields are kept: these responses carry account identifiers (email, name, org).
        try:
            v = rest.get_current_checkpoint_storage_usage()
            v = v.result() if hasattr(v, "result") else v
            out["checkpoint_storage_rate_usd_per_gb_month"] = getattr(v, "effective_rate_usd_per_gigabyte_month", None)
            out["checkpoint_count"] = sum(getattr(d, "checkpoint_count", 0) or 0 for d in getattr(v, "data", []) or [])
        except Exception as exc:
            out["storage_usage_error"] = type(exc).__name__
        return out
    step("billing_probe", billing, needs=("svc",))
    print(json.dumps([{k: s.get(k) for k in ("step", "ok", "error_type", "error_message")} for s in receipt["steps"]], indent=1))


if __name__ == "__main__":
    main()
