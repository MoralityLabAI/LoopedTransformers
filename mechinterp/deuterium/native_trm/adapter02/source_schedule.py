"""Run the complete paired optimizer schedule on excluded smoke source 61077.

The pinned upstream ``optimize`` owns every optimizer, batch and evaluation
step. A scoped adapter changes its model class and the PGD metric boundary.
Stock execution calls the original implementations. This module has no API
for a scientific teacher seed or a reserved data split.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import platform
import subprocess
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.deuterium.native_trm.fit_and_select import config
from mechinterp.deuterium.native_trm.adapter02.pass_indexed import (
    PassIndexedComponentModel, TARGET,
)
from mechinterp.deuterium.native_trm.adapter02 import pgd_shared
from param_decomp import run_param_decomp as upstream
from param_decomp.metrics import pgd_masked_recon_loss as pgd_metric
from param_decomp.metrics import pgd_utils
from param_decomp.models.batch_and_loss_fns import run_batch_first_element, recon_loss_kl
from param_decomp.models.component_model import ComponentModel, CIOutputs
from param_decomp.routing import AllLayersRouter

SCHEMA = "DEUTERIUM-NATIVE-02/source-schedule-v1"
SMOKE_SEED = 61077
SOURCE_N = 512
TOL = 2e-5
UPSTREAM_SHA = "2df0092d5be4d8e4748b3c66ef8325dedef17549"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return hashlib.sha256(header+tensor.numpy().tobytes()).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    text = "\n".join(f"{k}:{tensor_hash(v)}" for k,v in sorted(state.items()))
    return hashlib.sha256(text.encode()).hexdigest()


def vector(parameters, *, gradients=False) -> torch.Tensor:
    values = []
    for parameter in parameters:
        value = parameter.grad if gradients else parameter
        if value is None:
            raise RuntimeError("missing optimizer gradient")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("nonfinite optimizer value")
        values.append(value.detach().cpu().flatten().clone())
    return torch.cat(values)


def maximum_error(a, b) -> float:
    return float((a-b).abs().max())


def save_raw_trace(out: Path, snapshots: dict) -> dict:
    """Portable little-endian float32 vectors for an independent stdlib audit."""
    raw = bytearray()
    entries = []
    def append(phase, step, field, value):
        data = value.detach().cpu().numpy().astype("<f4",copy=False).tobytes()
        entries.append({"phase":phase,"step":step,"field":field,
                        "offset":len(raw),"count":value.numel(),
                        "sha256":hashlib.sha256(data).hexdigest()})
        raw.extend(data)
    for phase in ("warmup","main"):
        for step, update in enumerate(snapshots[phase]):
            for field in ("gradients","parameters"):
                append(phase,step,field,update[field])
    for field in ("terminal_gradients","terminal_parameters"):
        if field in snapshots:
            append("terminal",64,field,snapshots[field])
    path = out/"optimizer_trace.f32"
    path.write_bytes(raw)
    return {"file":path.name,"sha256":hashlib.sha256(raw).hexdigest(),"entries":entries}


def trajectory_comparison(reference, twin, stock, equal):
    errors,rows = {},[]
    for phase in ("warmup","main"):
        for step,(a,b) in enumerate(zip(reference[phase],twin[phase],strict=True)):
            row = {"phase":phase,"step":step}
            for field in ("parameters","gradients"):
                value = maximum_error(a[field],b[field])
                row[field] = value
                key = phase+"_"+field
                errors[key] = max(errors.get(key,0),value)
            if phase=="main":
                row["raw_losses"] = {name:abs(stock["losses"][step]["losses"][name]-
                    equal["losses"][step]["losses"][name]) for name in stock["losses"][step]["losses"]}
            rows.append(row)
    errors["terminal_gradients"] = maximum_error(reference["terminal_gradients"],twin["terminal_gradients"])
    errors["losses"] = max(abs(a["losses"][k]-b["losses"][k])
        for a,b in zip(stock["losses"],equal["losses"],strict=True) for k in a["losses"])
    return errors,{"tolerance":TOL,"step_indexing":"zero-based, before each main optimizer update",
                   "updates":rows,"first_main_update":rows[32]}


def run_arm(mode: str, teacher, inputs, cfg, out: Path):
    """Observe the pinned optimizer without replacing its training loop."""
    out.mkdir(parents=True, exist_ok=False)
    trace = {"mode": mode, "initial_state_sha256": None, "losses": [],
             "updates": [], "evaluations": [], "adversaries": [], "ci": []}
    snapshots = {"warmup": [], "main": []}
    created = []
    phase = {"value": "warmup"}
    original_optimizer = torch.optim.AdamW
    original_losses = upstream.compute_losses
    original_evaluate = upstream.evaluate
    original_pgd_update = pgd_metric.pgd_masked_recon_loss_update
    original_pgd_init = pgd_utils.get_pgd_init_tensor
    base_class = ComponentModel if mode == "stock" else PassIndexedComponentModel

    class ObservedModel(base_class):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)
            trace["initial_state_sha256"] = state_hash(self.state_dict())

        def calc_causal_importances(self, *args, **kwargs):
            ci = super().calc_causal_importances(*args, **kwargs)
            if mode == "last_repeat":
                ci = CIOutputs(**{
                    name: {TARGET: getattr(ci,name)[TARGET][-1:].expand(4,-1,-1,-1)}
                    for name in ("lower_leaky", "upper_leaky", "pre_sigmoid")
                })
            low = ci.lower_leaky[TARGET]
            trace["ci"].append({"shape": list(low.shape), "min": float(low.min().detach()),
                                "max": float(low.max().detach())})
            return ci

    class ObservedAdamW(original_optimizer):
        def __init__(self, params, *args, **kwargs):
            super().__init__(params, *args, **kwargs)
            self.phase_name = "warmup" if kwargs.get("weight_decay") == .1 else "main"

        def step(self, closure=None):
            parameters = [p for group in self.param_groups for p in group["params"]]
            gradients = vector(parameters, gradients=True)
            loss = None
            if self.phase_name == "warmup":
                with torch.no_grad():
                    loss = float(upstream.faithfulness_loss(created[0].calc_weight_deltas()))
            result = super().step(closure)
            params_after = vector(parameters)
            snapshots[self.phase_name].append({"gradients": gradients, "parameters": params_after})
            trace["updates"].append({
                "phase": self.phase_name, "update": len(snapshots[self.phase_name]),
                "lr": self.param_groups[0]["lr"], "weight_decay": self.param_groups[0]["weight_decay"],
                "gradient_norm": float(gradients.norm()), "gradient_max_abs": float(gradients.abs().max()),
                "gradient_sha256": tensor_hash(gradients), "parameter_sha256": tensor_hash(params_after),
                "warmup_faithfulness_before_update": loss,
            })
            return result

    def observed_losses(**kwargs):
        phase["value"] = "main"
        rng = tensor_hash(torch.get_rng_state())
        losses = original_losses(**kwargs)
        values = {one.classname: float(value.detach()) for one,value in losses.items()}
        trace["losses"].append({"step": len(trace["losses"]),
            "batch_sha256": tensor_hash(kwargs["batch"][0]), "rng_before": rng,
            "rng_after": tensor_hash(torch.get_rng_state()), "losses": values,
            "total": sum(one.coeff*float(value.detach()) for one,value in losses.items())})
        return losses

    def observed_evaluate(**kwargs):
        phase["value"] = "evaluation"
        result = original_evaluate(**kwargs)
        trace["evaluations"].append({"after_loss_step": len(trace["losses"])-1, "metrics": result})
        phase["value"] = "main"
        return result

    def shared_update(**kwargs):
        if mode == "stock":
            return original_pgd_update(**kwargs)
        if not isinstance(kwargs.pop("router"), AllLayersRouter):
            raise RuntimeError("source schedule requires the frozen all-layers router")
        kwargs["ci"] = kwargs["ci"][TARGET]
        return pgd_shared.pgd_pass_indexed_recon_loss_update(**kwargs)

    def observed_init(init, shape, device):
        value = original_pgd_init(init, shape, device=device)
        trace["adversaries"].append({"phase": phase["value"], "shape": list(shape),
                                     "initial_sha256": tensor_hash(value)})
        return value

    torch.manual_seed(SMOKE_SEED)
    frozen = copy.deepcopy(teacher)
    original_teacher_sha = state_hash(frozen.state_dict())
    train = DataLoader(TensorDataset(inputs), batch_size=cfg.batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(911000+SMOKE_SEED))
    evaluate = DataLoader(TensorDataset(inputs), batch_size=cfg.eval_batch_size, shuffle=False)
    dump(out/"config.json",cfg.model_dump(mode="json"))
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(upstream,"ComponentModel",ObservedModel))
            stack.enter_context(patch.object(torch.optim,"AdamW",ObservedAdamW))
            stack.enter_context(patch.object(upstream,"compute_losses",observed_losses))
            stack.enter_context(patch.object(upstream,"evaluate",observed_evaluate))
            stack.enter_context(patch.object(pgd_metric,"pgd_masked_recon_loss_update",shared_update))
            stack.enter_context(patch.object(pgd_utils,"get_pgd_init_tensor",observed_init))
            stack.enter_context(patch.object(pgd_shared,"get_pgd_init_tensor",observed_init))
            upstream.optimize(target_model=frozen, config=cfg, device="cpu",
                train_loader=train, eval_loader=evaluate, run_batch=run_batch_first_element,
                reconstruction_loss=recon_loss_kl, out_dir=out)
        if len(created) != 1 or state_hash(frozen.state_dict()) != original_teacher_sha:
            raise RuntimeError("frozen original teacher changed")
        optimized = list(created[0]._components.parameters())+list(created[0].ci_fn.parameters())
        snapshots["terminal_gradients"] = vector(optimized,gradients=True)
        snapshots["terminal_parameters"] = vector(optimized)
        trace["teacher_sha256"] = original_teacher_sha
        trace["terminal_state_sha256"] = state_hash(created[0].state_dict())
        trace["status"] = "COMPLETED"
    except Exception as exc:
        trace["status"] = "BLOCKED_UNMATCHED_OPTIMIZER"
        trace["error"] = {"type":type(exc).__name(),"message":str(exc),"traceback":traceback.format_exc()}
        raise
    finally:
        trace["runtime_seconds"] = time.perf_counter()-started
        torch.save(snapshots,out/"optimizer_trace.pt")
        trace["tensor_trace"] = save_raw_trace(out,snapshots)
        dump(out/"trace.json",trace)
    return trace,snapshots


def run(out: Path) -> dict:
    if out.exists():
        raise FileExistsError("source schedule output already exists; preserve its first receipt")
    torch.set_num_threads(2)
    root = Path(inspect.getfile(upstream)).resolve().parents[1]
    upstream_sha = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if upstream_sha != UPSTREAM_SHA:
        raise RuntimeError("pinned upstream revision mismatch")
    cfg = config(SMOKE_SEED,smoke=False)
    if (cfg.steps,cfg.faithfulness_warmup_steps,cfg.batch_size,cfg.sampling) != (64,32,8,"continuous"):
        raise RuntimeError("frozen optimizer schedule drift")
    minimality = next(c for c in cfg.loss_metric_configs if c.classname=="ImportanceMinimalityLoss")
    if minimality.beta != 0 or minimality.p_anneal_final_p is not None:
        raise RuntimeError("unregistered depth normalization or annealing")
    torch.manual_seed(SMOKE_SEED)
    teacher = TinyRecursiveModel(48,4).eval().requires_grad_(False)
    inputs,_ = fixed_data(SOURCE_N,SMOKE_SEED)
    record = {"schema":SCHEMA,"status":"RUNNING","source_seed":SMOKE_SEED,
        "source_n":SOURCE_N,"trained_teacher_used":False,"inputs_sha256":tensor_hash(inputs),
        "upstream_commit":upstream_sha,"torch":torch.__version__,"python":platform.python_version(),
        "tolerance":TOL,"reserved_teacher_roots_opened":[],"reserved_data_splits_opened":[],
        "scientific_evaluation":False,"arms":{},
        "gate_scope":"full-trajectory identity; stricter than protocol pointwise identity"}
    dump(out/"schedule_receipt.json",record)
    try:
        outputs = {}
        for mode in ("stock","last_repeat","native_pass"):
            outputs[mode] = run_arm(mode,teacher,inputs,cfg,out/mode)
            trace,_ = outputs[mode]
            counts = {name:sum(r["phase"]==name for r in trace["updates"])
                      for name in ("warmup","main")}
            if counts != {"warmup":32,"main":64} or len(trace["losses"]) != 65:
                raise RuntimeError(f"optimizer step counts changed: {mode}")
            if [e["after_loss_step"] for e in trace["evaluations"]] != [0,64]:
                raise RuntimeError("original evaluation schedule changed")
            if len(trace["adversaries"]) != 67 or any(a["shape"] != [8,81,17] for a in trace["adversaries"]):
                raise RuntimeError("shared PGD dimension or invocation budget changed")
            record["arms"][mode] = {"counts":counts,"main_loss_evaluations":len(trace["losses"]),
                "evaluation_steps":[e["after_loss_step"] for e in trace["evaluations"]],
                "adversary_draws":len(trace["adversaries"]),"runtime_seconds":trace["runtime_seconds"],
                "teacher_sha256":trace["teacher_sha256"],
                "initial_state_sha256":trace["initial_state_sha256"],
                "terminal_state_sha256":trace["terminal_state_sha256"]}
            dump(out/"schedule_receipt.json",record)
        stock,reference = outputs["stock"]
        equal,twin = outputs["last_repeat"]
        for mode,(trace,_) in outputs.items():
            if trace["initial_state_sha256"] != stock["initial_state_sha256"]:
                raise RuntimeError(f"initial factor or CI mismatch: {mode}")
            for key in ("batch_sha256","rng_before","rng_after"):
                if [r[key] for r in trace["losses"]] != [r[key] for r in stock["losses"]]:
                    raise RuntimeError(f"paired source/RNG schedule mismatch: {mode}:{key}")
            if trace["adversaries"] != stock["adversaries"]:
                raise RuntimeError(f"paired PGD initialization schedule mismatch: {mode}")
        errors,comparison = trajectory_comparison(reference,twin,stock,equal)
        dump(out/"trajectory_comparison.json",comparison)
        record["trajectory_comparison_sha256"] = hashlib.sha256((out/"trajectory_comparison.json").read_bytes()).hexdigest()
        record["forced_equal_max_abs_errors"] = errors
        if any(value > TOL for value in errors.values()):
            record["blocking_gate"] = "FULL_TRAJECTORY_IDENTITY_TOLERANCE"
            raise RuntimeError("forced-identical-mask full-schedule parity exceeded fixed tolerance")
        record["status"] = "SOURCE_ONLY_FULL_SCHEDULE_PASS"
    except Exception as exc:
        record["status"] = "BLOCKED_UNMATCHED_OPTIMIZER"
        record["error"] = {"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        raise
    finally:
        dump(out/"schedule_receipt.json",record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    fresh_output = not args.out.exists()
    try:
        result = run(args.out)
    except Exception:
        receipt = args.out/"schedule_receipt.json"
        if fresh_output and receipt.exists() and json.loads(receipt.read_text()).get("blocking_gate")=="FULL_TRAJECTORY_IDENTITY_TOLERANCE":
            print("DEUTERIUM_NATIVE02_SCHEDULE BLOCKED_UNMATCHED_OPTIMIZER",flush=True)
            raise SystemExit(2)
        raise
    print("DEUTERIUM_NATIVE02_SCHEDULE",result["status"],flush=True)


if __name__ == "__main__":
    main()
