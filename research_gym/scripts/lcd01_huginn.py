"""LCD-01 native Huginn pilot. From repository root:
python -m research_gym.scripts.lcd01_huginn --stage smoke
python -m research_gym.scripts.lcd01_huginn --stage calibrate --out /path/to/cal.json
python -m research_gym.scripts.lcd01_huginn --stage test --freeze /path/to/cal.json --out /path/to/test.json

No model download unless explicitly allowed. Calibration and held-out receipts are separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import torch

MODEL = "tomg-group-umd/huginn-0125"
REV = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
CAL_SEED, TEST_SEED = 2026112501, 2026112502
DEPTHS = (8, 16, 32)
FAMILIES = (1, 2, 3)
DOSES = (.25, .5, 1.)
REFS = (1, 6, 7)
METHODS = ("logits", "adaptive_logits", "hidden", "random_hidden", "orthogonal_logits", "temperature")
SCHEMA = "lcd01-v1"

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def configs(depth):
    base = [dict(method="base", depth=depth, ref=1, dose=0.)]
    for m in METHODS:
        refs = [r for r in REFS if r < depth] if m in ("hidden", "random_hidden") else [1]
        for r in refs:
            for w in DOSES:
                base.append(dict(method=m, depth=depth, ref=r, dose=w))
    return base

def ortho(v, axis):
    a = axis.float()
    v = v.float()
    return v - (v * a).sum(-1, keepdim=True) / a.square().sum(-1, keepdim=True).clamp_min(1e-20) * a

def random_direction(delta, seed):
    # Same norm for each token's contrast, orthogonal to the original direction.
    rng = torch.Generator(device=delta.device).manual_seed(seed)
    v = torch.randn(delta.shape, generator=rng, device=delta.device, dtype=torch.float32)
    v = ortho(v, delta)
    return v * (delta.float().norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True).clamp_min(1e-20))

def hidden_guidance(h, ref, dose, random_seed=None):
    d = h.float() - ref.float()
    if random_seed is not None:
        d = random_direction(d, random_seed)
    return (h.float() + dose * d).to(h.dtype)

def logit_guidance(z, ref, dose, method):
    z, ref = z.float(), ref.float()
    d = z - ref
    if method == "adaptive_logits":
        top = z.softmax(-1).topk(2).values
        dose = dose * (1. - top[..., :1] + top[..., 1:2])
    elif method == "orthogonal_logits":
        d = ortho(d - d.mean(-1, keepdim=True), z - z.mean(-1, keepdim=True))
    elif method == "temperature":
        # Positive scaling; cannot alter greedy rank. Calibration control only.
        return z * (1. + dose)
    return z + dose * d

def smoke():
    g = torch.Generator().manual_seed(42)
    h, r = (torch.randn(2, 3, 32, generator=g) for _ in range(2))
    d = h - r
    rd = random_direction(d, 19)
    assert torch.allclose(rd.norm(dim=-1), d.norm(dim=-1), atol=1e-5)
    assert torch.allclose((rd*d).sum(-1), torch.zeros(2,3), atol=2e-5)
    assert torch.equal(hidden_guidance(h, r, 0.), h)
    assert torch.equal(logit_guidance(h, r, .5, "temperature").argmax(-1), h.argmax(-1))
    assert torch.allclose(logit_guidance(h, r, .5, "logits"), 1.5*h-.5*r)
    assert torch.allclose(logit_guidance(h+7, r-9, .5, "orthogonal_logits").softmax(-1),
                          logit_guidance(h, r, .5, "orthogonal_logits").softmax(-1), atol=1e-6)
    print("LCD-01 tensor smoke PASS")

def items(seed, n):
    import numpy as np
    from research_gym.scripts.rmp_huginn_depth_sweep import make_item
    rng = np.random.default_rng(seed)
    out = []
    for hop in FAMILIES:
        seen = set()
        while len(seen) < n:
            prompt, answer = make_item(rng, hop, 3, "varchain")
            key = hashlib.sha256(prompt.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(id=key, family=hop, prompt=prompt, answer=answer))
    return out

def sync():
    torch.cuda.synchronize()

@torch.inference_mode()
def trajectory(model, ids, seed, depth, refs):
    torch.manual_seed(seed)
    e, idx = model.embed_inputs(ids)
    h = model.initialize_state(e, scale=1.)
    states = {}
    for t in range(1, depth+1):
        h, idx, _ = model.iterate_one_step(e, h, block_idx=idx, current_step=t-1)
        if t in refs or t == depth:
            states[t] = h.clone()
    return states

def read(model, h):
    return model.predict_from_latents(h.clone()).logits[:, -1].float()

@torch.inference_mode()
def score_from_states(model, states, cfg, seed):
    m, r, w, depth = (cfg[k] for k in ("method", "ref", "dose", "depth"))
    h = states[depth]
    if m == "base":
        return read(model, h), 1
    if m in ("hidden", "random_hidden"):
        return read(model, hidden_guidance(h, states[r], w, seed if m == "random_hidden" else None)), 1
    z = read(model, h)
    if m == "temperature":
        return logit_guidance(z, z, w, m), 1
    return logit_guidance(z, read(model, states[r]), w, m), 2

def row(z, cand, target):
    v = z[0, cand]
    if not torch.isfinite(v).all():
        raise ValueError("nonfinite candidate scores")
    lp = v.log_softmax(-1)
    return dict(pred=int(v.argmax()), correct=int(v.argmax()) == target, nll=-float(lp[target]),
                candidate_logits=v.cpu().tolist())

def summarize(rows):
    out = {}
    for family in FAMILIES:
        rr = [r for r in rows if r["family"] == family]
        if not rr:
            raise ValueError("missing family")
        out[str(family)] = dict(n=len(rr), accuracy=sum(r["correct"] for r in rr)/len(rr),
                                nll=sum(r["nll"] for r in rr)/len(rr))
    return dict(families=out, accuracy=sum(v["accuracy"] for v in out.values())/len(out),
                nll=sum(v["nll"] for v in out.values())/len(out))

def paired(rows, baseline):
    import numpy as np
    b = {r["id"]: r for r in baseline}
    if set(b) != {r["id"] for r in rows}:
        raise ValueError("unpaired rows")
    diffs = [np.array([int(r["correct"])-int(b[r["id"]]["correct"])
                       for r in rows if r["family"] == f]) for f in FAMILIES]
    rng = np.random.default_rng(2026112503)
    reps = np.mean([rng.choice(d, size=(5000,len(d)), replace=True).mean(1) for d in diffs], axis=0)
    return dict(delta=float(np.mean([d.mean() for d in diffs])),
                ci95=np.quantile(reps,[.025,.975]).tolist(),
                wrong_to_right=sum(int((d==1).sum()) for d in diffs),
                right_to_wrong=sum(int((d==-1).sum()) for d in diffs))

def key(c):
    return json.dumps(c,sort_keys=True,separators=(",",":"))

def run(args):
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import PreTrainedTokenizerFast
    from research_gym.scripts.rmp_huginn_depth_sweep import load_huginn, answer_tokens
    if not torch.cuda.is_available():
        raise RuntimeError("native pilot requires CUDA; use --stage smoke on CPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    local = Path(snapshot_download(MODEL, revision=REV, local_files_only=not args.allow_download))
    tok = PreTrainedTokenizerFast(tokenizer_file=str(local/"tokenizer.json"),
                                  bos_token="<|begin_text|>", add_bos_token=True)
    labels, token_ids = answer_tokens(tok, "varchain")
    assert len(set(token_ids)) == len(labels)
    cand = torch.tensor(token_ids, device="cuda")
    model = load_huginn(local, torch.device("cuda")).eval()
    cal = items(CAL_SEED,16)
    evaluation = items(TEST_SEED,48) if args.stage == "test" else None
    meta = dict(schema=SCHEMA, model=MODEL, revision=REV, torch=torch.__version__,
                transformers=transformers.__version__, device=torch.cuda.get_device_name(),
                dtype=str(next(model.parameters()).dtype), cal_sha=digest(cal),
                source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.stage == "test":
        freeze = json.loads(args.freeze.read_text())
        for k in ("schema","model","revision","cal_sha","source_sha","torch","transformers","dtype","device"):
            if freeze["meta"][k] != meta[k]:
                raise ValueError("freeze mismatch: "+k)
        if {r["id"] for r in cal} & {r["id"] for r in evaluation}:
            raise ValueError("split collision")
        selected = freeze["selected"]
        if freeze["freeze_sha"] != digest(selected):
            raise ValueError("freeze digest mismatch")
        expected = {(m,d) for d in DEPTHS for m in ("base",)+METHODS}
        if {(c["method"],c["depth"]) for c in selected} != expected or len(selected)!=len(expected):
            raise ValueError("incomplete freeze")
        if any(c not in configs(c["depth"]) for c in selected):
            raise ValueError("invalid frozen config")
        use = evaluation
        meta["eval_sha"] = digest(use)
        meta["freeze_sha"] = freeze["freeze_sha"]
    else:
        selected = [c for depth in DEPTHS for c in configs(depth)]
        use = cal
    # Admission check: manual recurrent path vs native forward on calibration only.
    parity = []
    with torch.inference_mode():
        for it in cal[:3]:
            ids = torch.tensor([tok.encode(it["prompt"])], device="cuda")
            assert int(ids[0,0]) == 65504
            for depth in DEPTHS:
                h = trajectory(model,ids,73,depth,())
                a = read(model,h[depth])
                torch.manual_seed(73)
                b = model(input_ids=ids,num_steps=depth,use_cache=False).logits[:,-1].float()
                err = float((a-b).abs().max())
                parity.append(dict(id=it["id"],depth=depth,max_abs=err,
                                   top1_equal=bool((a.argmax(-1)==b.argmax(-1)).all())))
                if not torch.isfinite(a).all() or not torch.isfinite(b).all() or err > .125 or not parity[-1]["top1_equal"]:
                    write(args.out, dict(meta=meta,status="BLOCKED_PARITY",parity=parity))
                    raise RuntimeError("manual recurrence parity failed; inspect receipt")
    all_rows = {key(c):[] for c in selected}
    for it in use:
        seed = int(it["id"][:8],16)
        ids = torch.tensor([tok.encode(it["prompt"])],device="cuda")
        target = labels.index(it["answer"])
        # Calibration shares trajectories; test times each arm independently in deterministic shuffled order.
        if args.stage == "calibrate":
            states = trajectory(model,ids,seed,max(DEPTHS),set(REFS)|set(DEPTHS))
        import random
        order = list(selected)
        random.Random(seed).shuffle(order)
        for c in order:
            if args.stage == "test":
                sync()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                refs = () if c["method"] in ("base","temperature") else (c["ref"],)
                states = trajectory(model,ids,seed,c["depth"],refs)
            with torch.inference_mode():
                z, passes = score_from_states(model,states,c,seed ^ 123456)
            if args.stage == "test":
                sync()
                ms = (time.perf_counter()-start)*1000
                peak = torch.cuda.max_memory_allocated()
            else:
                ms, peak = None,None
            record = row(z,cand,target)
            record.update(id=it["id"],family=it["family"],target=target,seed=seed,
                          tokens=ids.shape[-1],loops=c["depth"],coda_passes=passes,
                          latency_ms=ms,peak_bytes=peak)
            all_rows[key(c)].append(record)
            del z
        if len(all_rows[key(selected[0])]) % 8 == 0:
            print(args.stage,len(all_rows[key(selected[0])]),"/",len(use),flush=True)
            write(args.out,dict(meta=meta,status="IN_PROGRESS",parity=parity,rows=all_rows))
    summaries = {k:summarize(v) for k,v in all_rows.items()}
    if args.stage == "calibrate":
        chosen = []
        for d in DEPTHS:
            for m in ("base",)+METHODS:
                # Temperature selected by NLL; other arms use macro accuracy then NLL.
                options = [c for c in selected if c["depth"]==d and c["method"]==m]
                chosen.append(min(options,key=lambda c:(
                    0 if m=="temperature" else -summaries[key(c)]["accuracy"],
                    summaries[key(c)]["nll"],c["dose"],c["ref"])))
        write(args.out,dict(meta=meta,status="CALIBRATED",parity=parity,selected=chosen,
                           freeze_sha=digest(chosen),summaries=summaries,rows=all_rows))
    else:
        comparisons = {}
        for c in selected:
            k = key(c)
            same = key(dict(method="base",depth=c["depth"],ref=1,dose=0.))
            deep = key(dict(method="base",depth=min(32,2*c["depth"]),ref=1,dose=0.))
            rr = all_rows[k]
            def total(rows): return sum(r["latency_ms"] for r in rows)
            def layers(rows): return sum(r["tokens"]*(2+4*r["loops"]+2*r["coda_passes"]) for r in rows)
            comparisons[k] = dict(same_depth=paired(rr,all_rows[same]),deeper=paired(rr,all_rows[deep]),
                latency_ratio=total(rr)/total(all_rows[deep]),
                layer_token_ratio=layers(rr)/layers(all_rows[deep]),
                peak_bytes=max(r["peak_bytes"] for r in rr))
        write(args.out,dict(meta=meta,status="PILOT_COMPLETE",parity=parity,selected=selected,
                           summaries=summaries,comparisons=comparisons,rows=all_rows))

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("smoke","calibrate","test"),required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--freeze",type=Path)
    p.add_argument("--allow-download",action="store_true")
    a=p.parse_args()
    if a.stage=="smoke":
        smoke()
    else:
        if a.out is None or (a.stage=="test" and a.freeze is None):
            p.error("native stages need --out and test needs --freeze")
        if a.out.exists():
            p.error("output exists; preserve receipts and use a new path")
        run(a)
