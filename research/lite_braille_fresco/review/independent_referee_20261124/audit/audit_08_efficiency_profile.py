"""Audit 08 - What the reported efficiency numbers measure.

Reported (LBF_17_RESULTS.md section 3): rel_prov_opened 1,713.2 ms / episode, 150.3 MFLOPs /
decision and 4,096.9 KB "peak tensor"; sym_prov_readout 696.7 ms, 26.1 MFLOPs, 5,016.0 KB.

This script establishes, on the open calibration cohort with the archived checkpoints:

  1. INSTRUMENTATION IN THE TIMED REGION.  The registered runner times an episode around a loop
     that also (i) runs `probe_lbf17_decision` (eight forward passes with tensor capture and
     assertions instead of one) on the first two decisions of three seed-107 arms and (ii)
     runs one additional FlopCounterMode forward pass per unseen (arch, m0, m1, L) shape.
     We time the registered runner cold (as archived) against an uninstrumented loop.
  2. DECISION-LEVEL COST ON IDENTICAL INPUTS.  Episode times confound cost per decision with
     trajectory length, so each architecture is also timed on the same fixed set of frontier
     states (median of repeated, interleaved runs).
  3. FLOP COVERAGE.  FlopCounterMode counts matmul/linear/conv/einsum-style operators only.
     We report, per architecture, the share of profiled CPU self-time spent in operators it
     does not count (element-wise products, amax, tanh, masking, the trellis filter ...).
  4. MEMORY.  "Peak tensor KB" is `estimate_peak_tensor_bytes`, a closed-form expression, not
     a measurement.  We compare it with the peak of the profiler's allocation timeline.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from audit_03_replication_factorial import clean_episode  # noqa: E402

ARCHS = ("rel_prov_opened", "sym_prov_readout", "param_nonrecurrent", "compute_nonrecurrent",
         "lbf16_pos1_opened", "adj2_noprov_opened")


def sample_states(rg, m15, cases, steps=(3, 6)):
    out = []
    for c in cases:
        s = m15.FringeState.start(c)
        taken = 0
        for target in steps:
            while taken < target and s.live(128):
                (sd, ix), _, _ = m15.geom_mm_select(rg, c, s)
                m15.expand_node(rg, c, s, sd, ix)
                taken += 1
            if s.live(128) and len(s.fronts[0]) * len(s.fronts[1]) >= 4:
                out.append((c, s.clone_light()))
    return out


def measured_peak_bytes(model, c, s) -> int | None:
    """Peak of net CPU tensor allocations during one forward pass (profiler allocation events)."""
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
            with torch.no_grad():
                model(c, s, 128 - s.work, mode="normal")
        events = []
        for e in prof.profiler.kineto_results.events():
            if e.name() == "[memory]":
                events.append((e.start_ns() if hasattr(e, "start_ns") else e.start_us(), e.nbytes()))
        events.sort()
        cur = peak = 0
        for _, nb in events:
            cur += nb
            peak = max(peak, cur)
        return int(peak)
    except Exception as exc:  # pragma: no cover - profiler API differences
        return None


def op_time_split(model, c, s) -> dict:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.no_grad():
            for _ in range(3):
                model(c, s, 128 - s.work, mode="normal")
    # Operators whose arithmetic FlopCounterMode's formulas account for (matmul family and
    # convolutions, including the backend kernels that carry the convolution's self-time).
    counted_prefixes = ("aten::mm", "aten::matmul", "aten::bmm", "aten::addmm", "aten::baddbmm", "aten::linear",
                        "aten::conv", "aten::_convolution", "aten::convolution", "aten::einsum",
                        "aten::mkldnn_convolution", "aten::thnn_conv", "aten::_conv_depthwise", "aten::addbmm",
                        "aten::_nnpack_spatial_convolution", "aten::_slow_conv", "aten::slow_conv")
    tot = counted = 0.0
    top = defaultdict(float)
    for e in prof.key_averages():
        t = float(e.self_cpu_time_total)
        tot += t
        if e.key.startswith(counted_prefixes):
            counted += t
        top[e.key] += t
    ranked = sorted(top.items(), key=lambda kv: -kv[1])[:8]
    return {"self_cpu_time_share_in_flop_counted_operators": counted / max(tot, 1e-9),
            "self_cpu_time_share_uncounted": 1 - counted / max(tot, 1e-9),
            "top_operators_by_self_time_share": [(k, round(v / tot, 4)) for k, v in ranked]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pinned-sim", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    repo = Path(args.repo)
    scratch = common.scratch_dir(args.scratch)
    common.set_single_thread()
    rg, m15, m16, m17 = common.load_modules(repo, Path(args.pinned_sim))
    cases, _ = common.load_open_split(repo, rg, "calibration")
    cases = cases[:args.roots]
    models = {a: common.load_checkpoint(repo, m17, 107, a, scratch)[0] for a in ARCHS}
    recs = {a: common.load_checkpoint(repo, m17, 107, a, scratch)[1] for a in ARCHS}
    out: dict = {"torch": torch.__version__, "threads": torch.get_num_threads(), "roots": len(cases), "seed": 107}

    # ---- 1. instrumentation inside the registered timed region --------------------------------
    inst = {}
    for arch in ("rel_prov_opened", "sym_prov_readout"):
        m17._FLOP_CACHE.clear()
        reg_cold, reg_warm, clean = [], [], []
        for c in cases:
            pctx = ({"tag": f"107:{arch}", "checkpoint": recs[arch]["checkpoint_sha256"], "raw": [], "obs": []}
                    if arch == "rel_prov_opened" else None)   # exactly the archived probing rule for seed 107
            r = m17.run_lbf17_episode(rg, c, "learned", model=models[arch], arch=arch, mode="normal", probe_ctx=pctx)
            reg_cold.append(r["wall_clock_ms"])
        for c in cases:   # FLOP cache now warm, no probe
            r = m17.run_lbf17_episode(rg, c, "learned", model=models[arch], arch=arch, mode="normal", probe_ctx=None)
            reg_warm.append(r["wall_clock_ms"])
        for c in cases:
            clean.append(clean_episode(rg, m15, c, models[arch], arch, "normal", 128)["wall_clock_ms_clean"])
        inst[arch] = {
            "registered_runner_as_archived_ms_per_episode": st.mean(reg_cold),
            "registered_runner_warm_flop_cache_no_probe_ms": st.mean(reg_warm),
            "uninstrumented_loop_ms": st.mean(clean),
            "instrumentation_overhead_factor": st.mean(reg_cold) / st.mean(clean),
        }
    inst["ratio_rel_prov_over_sym_prov_as_archived_protocol"] = (
        inst["rel_prov_opened"]["registered_runner_as_archived_ms_per_episode"]
        / inst["sym_prov_readout"]["registered_runner_as_archived_ms_per_episode"])
    inst["ratio_rel_prov_over_sym_prov_uninstrumented"] = (
        inst["rel_prov_opened"]["uninstrumented_loop_ms"] / inst["sym_prov_readout"]["uninstrumented_loop_ms"])
    inst["reported_ratio_in_results"] = 1713.2 / 696.7
    out["instrumentation"] = inst

    # ---- 2. decision cost on identical frontier states ----------------------------------------
    states = sample_states(rg, m15, cases)
    out["fixed_states"] = {"count": len(states),
                           "mean_m0": st.mean(len(s.fronts[0]) for _, s in states),
                           "mean_m1": st.mean(len(s.fronts[1]) for _, s in states)}
    per = {a: [] for a in ARCHS}
    for a in ARCHS:  # warm-up
        with torch.no_grad():
            models[a](states[0][0], states[0][1], 100, mode="normal")
    for _ in range(args.repeats):
        for a in ARCHS:
            t0 = time.perf_counter()
            with torch.no_grad():
                for c, s in states:
                    models[a](c, s, 128 - s.work, mode="normal")
            per[a].append((time.perf_counter() - t0) * 1000.0 / len(states))
    flops = {}
    for a in ARCHS:
        tot = 0
        for c, s in states:
            fc = FlopCounterMode(display=False)
            with torch.no_grad(), fc:
                models[a](c, s, 128 - s.work, mode="normal")
            tot += fc.get_total_flops()
        flops[a] = tot / len(states)
    ref_ms, ref_fl = st.median(per["sym_prov_readout"]), flops["sym_prov_readout"]
    out["decision_cost_identical_states"] = {
        a: {"median_ms_per_decision": st.median(per[a]), "min_ms": min(per[a]), "max_ms": max(per[a]),
            "flopcounter_mflops_per_decision": flops[a] / 1e6,
            "time_ratio_vs_sym_prov_readout": st.median(per[a]) / ref_ms,
            "flop_ratio_vs_sym_prov_readout": flops[a] / ref_fl,
            "ns_per_counted_flop": 1e6 * st.median(per[a]) / flops[a]}
        for a in ARCHS}

    # ---- 3. FLOP coverage and 4. memory ---------------------------------------------------------
    big = max(states, key=lambda cs: len(cs[1].fronts[0]) * len(cs[1].fronts[1]))
    mid = sorted(states, key=lambda cs: len(cs[1].fronts[0]) * len(cs[1].fronts[1]))[len(states) // 2]
    cov, mem = {}, {}
    for a in ARCHS:
        cov[a] = op_time_split(models[a], mid[0], mid[1])
        rows = []
        for label, (c, s) in (("median_state", mid), ("largest_state", big)):
            m0, m1, L = len(s.fronts[0]), len(s.fronts[1]), len(c.source)
            est = m17.estimate_peak_tensor_bytes(a, m0, m1, L)
            meas = measured_peak_bytes(models[a], c, s)
            rows.append({"state": label, "m0": m0, "m1": m1, "L": L, "registered_estimate_bytes": est,
                         "profiler_peak_allocated_bytes": meas,
                         "measured_over_estimate": None if not meas else meas / est})
        mem[a] = rows
    # ---- 5. where a decision's time goes (what a cache could and could not remove) --------------
    model = models["rel_prov_opened"]
    groups = {
        "word_level_cells_and_gates (cacheable per word)": [model.pos_cell, model.bi_cell, model.env_gate, model.bi_gate],
        "pair_ribbon_in_and_conv (cacheable per pair and pass)": [model.ribbon_in, model.ribbon_conv],
        "frontier_level_pair_in_prelude_axial_heads (not cacheable)": [
            model.pair_in, model.pair_prelude, model.fresco_block, model.fresco_gate, model.pot_head, model.node_head],
    }
    acc_t = {g: 0.0 for g in groups}
    stack = []
    handles = []
    for g, mods in groups.items():
        for mod in mods:
            handles.append(mod.register_forward_pre_hook(lambda m, i, g=g: stack.append((g, time.perf_counter()))))
            def post(m, i, o, g=g):
                g0, t0 = stack.pop()
                acc_t[g0] += time.perf_counter() - t0
            handles.append(mod.register_forward_hook(post))
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(3):
            for c, s in states:
                model(c, s, 128 - s.work, mode="normal")
    total = time.perf_counter() - t0
    for h in handles:
        h.remove()
    split = {g: v / total for g, v in acc_t.items()}
    split["remainder_pairwise_overlap_construction_and_glue (mostly cacheable per pair)"] = 1 - sum(split.values())
    out["decision_time_split_rel_prov_opened"] = split
    out["flop_counter_coverage"] = cov
    out["memory"] = mem
    out["memory_note"] = "registered 'peak tensor KB' = max over decisions of estimate_peak_tensor_bytes(arch, m0, m1, L)"
    common.write_json(Path(args.out), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
