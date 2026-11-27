"""Remote Inkling-2-Mini sampling and LoRA training through Tinker (free tier only).

Two sampling routes:
  * OpenAI-compatible chat route (audit-01 runner): exposes ``reasoning_effort``; returns text
    and usage, not sampled token ids/logprobs.
  * Native SDK route: returns token ids and sampling logprobs, which RL losses need; the
    installed SDK's SamplingParams has no reasoning-effort field.
Training uses the native SDK only. Every entry point refuses any model id other than MODEL.
The credential is read from the environment and never logged, stored or echoed.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from . import resources  # noqa: F401  (sets CUDA_VISIBLE_DEVICES=-1 before SDK import)

MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
DEFAULT_CONTEXT = 12_288
CONTEXT_CEILING = 16_384
DEFAULT_MAX_TOKENS = 512
NATIVE_LOSSES = ("cross_entropy", "importance_sampling", "ppo", "cispo", "dro", "gspo")


class PaidModelBlocked(ValueError):
    pass


def guard_model(model: str) -> str:
    if model != MODEL:
        raise PaidModelBlocked("only the documented free-tier model id is allowed")
    return model


def guard_context(prompt_tokens: int, max_tokens: int, limit: int = DEFAULT_CONTEXT) -> None:
    if limit > CONTEXT_CEILING:
        raise ValueError("context limit above strict ceiling")
    if prompt_tokens + max_tokens > limit:
        raise ValueError(f"prompt {prompt_tokens} + generation {max_tokens} exceeds {limit}")


def credential_present() -> bool:
    return bool(os.environ.get("TINKER_API_KEY"))


def sdk_capabilities() -> dict:
    """Static inspection of the installed SDK. No network, no credential."""
    try:
        import tinker
        from tinker.types import LossFnType  # noqa: F401
        from tinker.types._pydantic_types.sampling_params import SamplingParams
        import typing
        losses = list(typing.get_args(tinker.types.LossFnType))
        sp_fields = sorted(SamplingParams.model_fields)
        from tinker.lib.public_interfaces.training_client import TrainingClient
        methods = sorted(m for m in ("forward", "forward_backward", "forward_backward_custom", "optim_step",
                                     "save_state", "load_state", "load_state_with_optimizer",
                                     "save_weights_for_sampler", "save_weights_and_get_sampling_client")
                         if hasattr(TrainingClient, m))
        return {"installed": True, "version": tinker.__version__, "loss_fns": losses,
                "sampling_param_fields": sp_fields,
                "native_reasoning_effort": "reasoning_effort" in sp_fields,
                "training_methods": methods,
                "tensor_access": "logprobs and loss outputs only; no activations, gradients or "
                                 "adapter weight tensors are returned by the documented client",
                "custom_loss": "forward_backward_custom (client-side torch on returned logprobs)"}
    except ImportError as exc:
        return {"installed": False, "error": type(exc).__name__}


# ---- receipts --------------------------------------------------------------------------
@dataclass
class Usage:
    sample_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    train_requests: int = 0
    train_tokens: int = 0
    optim_steps: int = 0
    checkpoints_saved: int = 0
    paid_spend_usd: float = 0.0  # free model only; any non-free id raises before a request


# ---- OpenAI-compatible sampling (effort-controlled) -------------------------------------
class OAISampler:
    def __init__(self, usage: Usage, transport=None, timeout: float = 120):
        from research_gym.ink_fp_audit_01 import tinker_remote as r01
        guard_model(r01.MODEL)
        self.r01, self.usage, self.timeout = r01, usage, timeout
        self.transport = transport or r01.sample

    def sample(self, prompt: str, effort: str, max_tokens: int, key: str) -> dict:
        payload = self.r01.request_payload(prompt, effort, max_tokens)
        guard_model(payload["model"])
        t0 = time.monotonic()
        reply = self.transport(key, payload, self.timeout)
        self.usage.sample_requests += 1
        u = reply.get("usage") or {}
        self.usage.prompt_tokens += int(u.get("prompt_tokens") or 0)
        self.usage.completion_tokens += int(u.get("completion_tokens") or 0)
        choice = reply["choices"][0]
        return {"text": choice["message"].get("content") or "", "finish_reason": choice.get("finish_reason"),
                "usage": u, "response_model": reply.get("model"), "response_id": reply.get("id"),
                "seconds": time.monotonic() - t0, "route": "oai", "effort": effort,
                "tokens": None, "logprobs": None,
                "completion_tokens": int(u.get("completion_tokens") or 0)}


# ---- Native SDK ---------------------------------------------------------------------------
class NativeTinker:
    """Thin wrapper over the native SDK. Constructed lazily; one remote request at a time."""

    def __init__(self, usage: Usage, rank: int = 8, seed: int = 0):
        if not credential_present():
            raise RuntimeError("TINKER_API_KEY not present in environment")
        import tinker
        from tinker import types
        self.tinker, self.types, self.usage = tinker, types, usage
        self.service = tinker.ServiceClient()
        self.rank, self.seed = rank, seed
        self.training = None
        self.sampler = None
        self.tokenizer = None

    def server_models(self) -> list[str]:
        caps = self.service.get_server_capabilities()
        return [m.model_name for m in getattr(caps, "supported_models", [])]

    def open_sampler(self, model_path: str | None = None):
        if model_path is None:
            self.sampler = self.service.create_sampling_client(base_model=guard_model(MODEL))
        else:
            self.sampler = self.service.create_sampling_client(model_path=model_path)
            if self.sampler.get_base_model() != MODEL:
                raise PaidModelBlocked("sampler base model is not the free model")
        self.tokenizer = self.sampler.get_tokenizer()
        return self.sampler

    def encode_chat(self, prompt: str) -> list[int]:
        msgs = [{"role": "user", "content": prompt}]
        return list(self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))

    def sample(self, prompt: str, n: int, max_tokens: int = DEFAULT_MAX_TOKENS,
               temperature: float = 1.0, seed: int | None = None) -> list[dict]:
        t = self.types
        ids = self.encode_chat(prompt)
        guard_context(len(ids), max_tokens)
        params = t.SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed)
        res = self.sampler.sample(prompt=t.ModelInput.from_ints(ids), num_samples=n,
                                  sampling_params=params).result()
        self.usage.sample_requests += 1
        self.usage.prompt_tokens += len(ids) * n
        out = []
        for seq in res.sequences:
            toks = list(seq.tokens)
            self.usage.completion_tokens += len(toks)
            out.append({"text": self.tokenizer.decode(toks, skip_special_tokens=True),
                        "tokens": toks, "logprobs": list(seq.logprobs or []),
                        "prompt_tokens": ids, "finish_reason": str(getattr(seq, "stop_reason", "")),
                        "completion_tokens": len(toks), "route": "native", "effort": "native_default"})
        return out

    # training -------------------------------------------------------------------------
    def open_training(self, resume_state: str | None = None):
        if resume_state:
            self.training = self.service.create_training_client_from_state_with_optimizer(resume_state)
        else:
            self.training = self.service.create_lora_training_client(
                base_model=guard_model(MODEL), rank=self.rank, seed=self.seed)
        info = self.training.get_info()
        if getattr(info, "model_name", MODEL) != MODEL:
            raise PaidModelBlocked("training client is not on the free model")
        return self.training

    def rl_datum(self, prompt_ids: list[int], tokens: list[int], logprobs: list[float], advantage: float):
        t = self.types
        full = prompt_ids + tokens
        n_prompt = len(prompt_ids)
        target = full[1:]
        lp = [0.0] * (n_prompt - 1) + list(logprobs)
        adv = [0.0] * (n_prompt - 1) + [advantage] * len(tokens)
        return t.Datum(model_input=t.ModelInput.from_ints(full[:-1]),
                       loss_fn_inputs={"target_tokens": t.TensorData(data=target, dtype="int64", shape=[len(target)]),
                                       "logprobs": t.TensorData(data=lp, dtype="float32", shape=[len(lp)]),
                                       "advantages": t.TensorData(data=adv, dtype="float32", shape=[len(adv)])})

    def sft_datum(self, prompt_ids: list[int], tokens: list[int]):
        t = self.types
        full = prompt_ids + tokens
        weights = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(tokens)
        target = full[1:]
        return t.Datum(model_input=t.ModelInput.from_ints(full[:-1]),
                       loss_fn_inputs={"target_tokens": t.TensorData(data=target, dtype="int64", shape=[len(target)]),
                                       "weights": t.TensorData(data=weights, dtype="float32", shape=[len(weights)])})

    def train_step(self, datums: list, loss_fn: str, lr: float, loss_fn_config: dict | None = None) -> dict:
        if loss_fn not in NATIVE_LOSSES:
            raise ValueError("unsupported loss")
        fb = self.training.forward_backward(datums, loss_fn, loss_fn_config).result()
        self.training.optim_step(self.types.AdamParams(learning_rate=lr)).result()
        n_tok = sum(d.model_input.length for d in datums)
        self.usage.train_requests += 1
        self.usage.train_tokens += n_tok
        self.usage.optim_steps += 1
        return {"loss_fn": loss_fn, "tokens": n_tok, "metrics": dict(getattr(fb, "metrics", {}) or {})}

    def save(self, name: str) -> dict:
        state = self.training.save_state(name).result()
        sampler = self.training.save_weights_for_sampler(name + "-sampler").result()
        self.usage.checkpoints_saved += 1
        return {"state_path": state.path, "sampler_path": sampler.path}


# ---- engine backend ---------------------------------------------------------------------
class TinkerBackend:
    """Engine protocol over Tinker.

    Effort-controlled samples come from the OpenAI-compatible route (base model only until a
    live check shows that route accepts trained-adapter paths). Training items are re-tokenised
    and scored under the current sampler with compute_logprobs to obtain the behaviour logprobs
    that importance_sampling / gspo need. Evaluation of a candidate uses the native sampler on
    its saved weights at temperature 0, so predecessor and candidate share one route.
    """

    simulated = False

    def __init__(self, usage: Usage, key_env: str = "TINKER_API_KEY", rank: int = 8, seed: int = 0,
                 max_tokens: int = DEFAULT_MAX_TOKENS, native=None, oai=None):
        self.key = os.environ.get(key_env)
        if not self.key:
            raise RuntimeError(f"{key_env} not present in environment")
        self.usage, self.max_tokens = usage, max_tokens
        self.native = native or NativeTinker(usage, rank=rank, seed=seed)
        self.oai = oai or OAISampler(usage)
        self.native.open_training()
        self.native.open_sampler()
        self.paths = {"base": None}
        self.active = "base"

    def efforts(self):
        return ("low", "medium", "high") if self.active == "base" else ("native_default",)

    def sample(self, task, effort, n, seed):
        if effort == "native_default":
            return self.native.sample(task.prompt, n, self.max_tokens, temperature=1.0, seed=seed)
        return [self.oai.sample(task.prompt, effort, self.max_tokens, self.key) for _ in range(n)]

    def _tokens(self, task, s):
        if s.get("tokens"):
            return s["prompt_tokens"], s["tokens"], s["logprobs"]
        prompt_ids = self.native.encode_chat(task.prompt)
        comp = list(self.native.tokenizer.encode(s["text"], add_special_tokens=False))
        guard_context(len(prompt_ids), len(comp))
        t = self.native.types
        lp = self.native.sampler.compute_logprobs(t.ModelInput.from_ints(prompt_ids + comp)).result()
        self.usage.sample_requests += 1
        return prompt_ids, comp, [float(x) for x in lp[len(prompt_ids):]]

    def train(self, arm, items):
        datums = []
        for it in items:
            p, c, lp = self._tokens(it["task"], it["sample"])
            if arm.method == "sft":
                datums.append(self.native.sft_datum(p, c))
            else:
                datums.append(self.native.rl_datum(p, c, lp, it["advantage"]))
        return self.native.train_step(datums, arm.loss_fn, arm.lr)

    def save(self, tag):
        paths = self.native.save(tag)
        self.paths[tag] = paths
        return tag

    def checkpoint_meta(self, tag):
        return dict(self.paths[tag])  # tinker:// state and sampler paths; no credentials

    def adopt(self, tag, meta):
        self.paths[tag] = dict(meta)

    def restore(self, ckpt):
        if ckpt == self.active:
            return
        if ckpt == "base":
            self.native.open_training()
            self.native.open_sampler()
        else:
            self.native.open_training(resume_state=self.paths[ckpt]["state_path"])
            self.native.open_sampler(self.paths[ckpt]["sampler_path"])
        self.active = ckpt

    def evaluate(self, tasks, ckpt):
        from .curriculum import verify
        prev = self.active
        self.restore(ckpt)
        out = []
        for t in tasks:
            s = self.native.sample(t.prompt, 1, self.max_tokens, temperature=0.0, seed=0)[0]
            v = verify(t, s["text"])
            out.append({"task_id": t.task_id, "family": t.family, "correct": v["correct"],
                        "completed": v["completed"], "repetition": v["repetition"],
                        "completion_tokens": s["completion_tokens"]})
        self.restore(prev)
        return out
