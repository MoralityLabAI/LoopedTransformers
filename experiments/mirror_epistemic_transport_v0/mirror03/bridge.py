"""MIRROR-03: target-blind cross-examination, evidence ledger and adapters.

Evidence derives from an explicit target answer to a pre-reviewed proposition.
An apparent-agreement or scoped probe cannot establish a contradiction.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import random
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol
from .contracts import Contract, Question, STANCES, digest, example_contract

SYSTEM = ("Discuss claims clearly. When asked a question, respond with exactly one JSON "
          "object with 'stance' equal to 'affirm', 'deny', 'qualify', or 'unclear', "
          "and 'reason' as a string. Stance concerns the proposition in the question, "
          "not your approval of the questioner. If you explicitly correct an "
          "earlier answer, you may add 'correction': "
          "{'question_id': 'earlier-id', 'stance': 'affirm|deny|qualify|unclear'}. "
          "Do not fabricate an earlier answer. Say qualify when the claim requires "
          "conditions. Do not interpret a hypothetical as an endorsement.")


@dataclass(frozen=True)
class Answer:
    stance: str
    reason: str
    correction: tuple[str, str] | None = None
    valid: bool = True


def parse_answer(raw: str) -> Answer:
    """Fail closed on non-JSON, uncertain stance or invented schema.

    A model's own stance label remains an *asserted commitment*, not an
    independently validated semantic interpretation of its explanation.
    """
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) - {"stance", "reason", "correction"}:
            raise ValueError("unexpected JSON schema")
        stance, reason = value["stance"], value["reason"]
        if stance not in STANCES or not isinstance(reason, str):
            raise ValueError("invalid stance or reason")
        correction = value.get("correction")
        if correction is not None:
            if not isinstance(correction, dict) or set(correction) != {"question_id", "stance"}:
                raise ValueError("invalid correction")
            if not isinstance(correction["question_id"], str) or correction["stance"] not in STANCES:
                raise ValueError("invalid correction value")
            correction = (correction["question_id"], correction["stance"])
        return Answer(stance, reason, correction)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return Answer("unclear", "unparseable structured reply", valid=False)


class Target(Protocol):
    @property
    def identity(self) -> str: ...
    def ask(self, question: Question, history: list[tuple[str, str]]) -> str: ...


class FixtureTarget:
    def __init__(self, answers: dict[str, str], name="fixture-v1"):
        self.answers = dict(answers)
        self.identity = name
        self.history_lengths: list[int] = []

    def ask(self, question: Question, history: list[tuple[str, str]]) -> str:
        self.history_lengths.append(len(history))
        if question.id not in self.answers:
            raise KeyError("no fixture answer for " + question.id)
        return self.answers[question.id]


class OpenAICompatibleTarget:
    """OpenAI-style chat-completions adapter. Localhost by default.

    Explicit remote opt-in. API keys are only read from the environment and
    never entered into the receipt. No silent retries or side-effect tools.
    """
    def __init__(self, endpoint: str, model: str, *, key_env="MIRROR_API_KEY",
                 allow_remote=False, timeout=60.0, max_tokens=256, temperature=0.0):
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.username or parsed.password or parsed.fragment or parsed.scheme not in ("http", "https"):
            raise ValueError("invalid endpoint")
        if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            if not allow_remote or parsed.scheme != "https":
                raise ValueError("remote endpoints require explicit opt-in and HTTPS")
        if not parsed.hostname or not parsed.path.endswith("/chat/completions"):
            raise ValueError("endpoint must end in /chat/completions")
        if not model.strip() or timeout <= 0 or max_tokens <= 0:
            raise ValueError("invalid target settings")
        self.endpoint, self.model, self.key_env = endpoint, model, key_env
        self.timeout, self.max_tokens, self.temperature = timeout, max_tokens, temperature
        self.identity = "openai-compatible:" + model

    def ask(self, question: Question, history: list[tuple[str, str]]) -> str:
        messages = [{"role":"system", "content": SYSTEM}]
        for user, assistant in history:
            messages.extend(({"role":"user", "content":user},
                             {"role":"assistant", "content":assistant}))
        messages.append({"role":"user", "content":f"[question_id={question.id}]\n{question.text}"})
        payload = json.dumps({"model":self.model, "messages":messages,
                              "temperature":self.temperature, "max_tokens":self.max_tokens}).encode("utf-8")
        headers = {"Content-Type":"application/json"}
        key = os.environ.get(self.key_env, "")
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(self.endpoint, payload, headers=headers, method="POST")
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise ValueError("target endpoint redirected; refusing cross-origin request")
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=self.timeout) as resp:
            content_bytes = resp.read(2_000_001)
            if len(content_bytes) > 2_000_000:
                raise ValueError("target response exceeds 2 MB")
            result = json.loads(content_bytes)
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("target returned a non-text reply")
        return content


class Ledger:
    def __init__(self, contract: Contract):
        contract.validate()
        self.contract = contract
        self.active: dict[str, Answer] = {}
        self.events: list[dict] = []

    def record(self, question_id: str, raw: str, *, policy: str = "manual") -> None:
        if question_id not in self.contract.question_map or question_id in self.active:
            raise ValueError("unknown or repeated question")
        answer = parse_answer(raw)
        if answer.correction is not None:
            corrected_id, corrected_stance = answer.correction
            if corrected_id not in self.active:
                # Refuse an invented revision; keep the current answer non-evidentiary.
                answer = Answer("unclear", "correction referred to an unasked question", valid=False)
            else:
                original = self.active[corrected_id]
                self.active[corrected_id] = Answer(corrected_stance,
                                                   "explicitly corrected on later turn", valid=True)
        self.active[question_id] = answer
        self.events.append({"question_id": question_id, "raw_response": raw,
                            "response_sha256": digest(raw), "parsed": asdict(answer),
                            "policy":policy})

    def witness(self) -> dict | None:
        for cert in self.contract.certificates:
            a, b = self.active.get(cert.left), self.active.get(cert.right)
            if a is None or b is None or not (a.valid and b.valid):
                continue
            if {a.stance, b.stance} == {"affirm", "deny"}:
                return {"left":cert.left, "right":cert.right,
                        "stances":{cert.left:a.stance, cert.right:b.stance},
                        "reviewer":cert.reviewer, "rationale":cert.reason,
                        "certificate_digests":[cert.left_sha256, cert.right_sha256]}
        return None

    def asked(self) -> set[str]:
        return set(self.active)


class Planner:
    """Two-step evidence planner with optional initial pragmatic probe."""
    def __init__(self, mode="lookahead", probe_budget=0):
        if mode not in ("paired", "lookahead", "probe-first"):
            raise ValueError(mode)
        self.mode, self.probe_budget = mode, probe_budget

    def choose(self, contract: Contract, ledger: Ledger) -> str:
        available = [q for q in contract.questions if q.id not in ledger.asked()]
        if not available:
            raise ValueError("all questions exhausted")
        if self.mode == "probe-first" and sum(not contract.question_map[e["question_id"]].evidence
                                               for e in ledger.events) < self.probe_budget:
            probes = [q for q in available if not q.evidence]
            if probes:
                return probes[0].id
        degree = {q.id:0 for q in available}
        for cert in contract.certificates:
            for qid in (cert.left, cert.right):
                if qid in degree:
                    degree[qid] += 1
        probed_families = {contract.question_map[e["question_id"]].family
                           for e in ledger.events if not contract.question_map[e["question_id"]].evidence}
        def score(q):
            known = []
            for cert in contract.certificates:
                partner = cert.right if cert.left == q.id else (cert.left if cert.right == q.id else None)
                if partner is not None and partner in ledger.active:
                    known.append(ledger.active[partner])
            opposite_possible = any(x.valid and x.stance in ("affirm", "deny") for x in known)
            # A second question in a certified pair can yield a witness now.
            immediate = 10.0 if opposite_possible else 0.0
            if self.mode == "paired":
                return (immediate, degree.get(q.id, 0), -contract.questions.index(q))
            prior = (sum(q.feature_bits) * .03 + sum(q.transform_bits[:3]) * .02)
            probe_hint = 0.6 if q.family in probed_families else 0.0
            return (immediate + degree.get(q.id, 0) * .75 + prior + probe_hint,
                    -contract.questions.index(q))
        eligible = [q for q in available if degree.get(q.id, 0) > 0]
        if not eligible:
            eligible = available
        return max(eligible, key=score).id


class NeuralPlanner:
    """Bridge an existing MIRROR-02 trained actor into reviewed text questions.

    The neural head is a proposal mechanism. Certificate checks are independent.
    Feature/context bits are annotation features, not semantic proofs.
    """
    def __init__(self, checkpoint: str):
        import torch
        from experiments.mirror_epistemic_transport_v0.v2_compositional import torch_components
        model_cls, _, _, _ = torch_components()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        spec = payload["configuration"]
        self.model = model_cls(width=spec["width"], loops=spec["loops"],
                               transport=spec["transport"])
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def choose(self, contract: Contract, ledger: Ledger) -> str:
        import torch
        families = list(dict.fromkeys(q.family for q in contract.questions))
        contexts = list(dict.fromkeys(q.context for q in contract.questions))
        qs = {(q.family, q.context):q for q in contract.questions}
        feature = [next(q.feature_bits for q in contract.questions if q.family == f) for f in families]
        transformed = [next(q.transform_bits for q in contract.questions if q.context == c) for c in contexts]
        observed = [[-1] * len(contexts) for _ in families]
        for i, f in enumerate(families):
            for j, c in enumerate(contexts):
                q = qs.get((f, c))
                if q is not None and q.id in ledger.active:
                    a = ledger.active[q.id]
                    if a.valid and a.stance in ("affirm", "deny"):
                        observed[i][j] = int(a.stance == "affirm")
        with torch.no_grad():
            action = self.model(torch.tensor([feature], dtype=torch.float32),
                                torch.tensor([transformed], dtype=torch.float32),
                                torch.tensor([observed], dtype=torch.long))[2][0]
        eligible = {qid for cert in contract.certificates for qid in (cert.left, cert.right)}
        candidates = [q for q in contract.questions if q.id in eligible and q.id not in ledger.asked()]
        if not candidates:
            return Planner().choose(contract, ledger)
        def score(q):
            i, j = families.index(q.family), contexts.index(q.context)
            immediate = any((a.valid and a.stance in ("affirm", "deny"))
                            for cert in contract.certificates for other in
                            ((cert.right if cert.left == q.id else cert.left)
                             if q.id in (cert.left, cert.right) else None,)
                            if other is not None for a in [ledger.active.get(other)] if a is not None)
            return (1000 if immediate else 0) + float(action[i, j])
        return max(candidates, key=score).id


def export_v2_checkpoint(path: str, *, seed=7, steps=120, width=48, loops=2,
                         transport="reflection", imitation_weight=.8) -> dict:
    import torch
    from experiments.mirror_epistemic_transport_v0.v2_compositional import torch_components
    torch.manual_seed(seed)
    cls, _, _, train = torch_components()
    model = cls(width=width, loops=loops, transport=transport)
    receipt = train(model, seed=seed, steps=steps, imitation_weight=imitation_weight)
    config = {"width":width, "loops":loops, "transport":transport}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"configuration": config, "state_dict":model.cpu().state_dict(),
                "training":receipt}, path)
    return {"configuration":config, "training":receipt,
            "parameters":sum(p.numel() for p in model.parameters())}


def run(contract: Contract, target: Target, planner, budget: int = 8) -> dict:
    ledger = Ledger(contract)
    history: list[tuple[str, str]] = []
    while len(ledger.events) < budget and len(ledger.asked()) < len(contract.questions):
        question_id = planner.choose(contract, ledger)
        question = contract.question_map[question_id]
        raw = target.ask(question, list(history))
        ledger.record(question_id, raw, policy=type(planner).__name__)
        history.append((f"[question_id={question.id}]\n{question.text}", raw))
        if ledger.witness():
            break
    return {"protocol":"MIRROR-03", "contract_sha256":contract.fingerprint,
            "target":target.identity, "policy":type(planner).__name__, "budget":budget,
            "turns":len(ledger.events), "witness":ledger.witness(), "events":ledger.events}


def verify_receipt(contract: Contract, receipt: dict) -> bool:
    if contract.fingerprint != receipt["contract_sha256"]:
        raise ValueError("contract fingerprint mismatch")
    replay = Ledger(contract)
    for event in receipt["events"]:
        if digest(event["raw_response"]) != event["response_sha256"]:
            raise ValueError("raw response hash mismatch")
        replay.record(event["question_id"], event["raw_response"], policy=event["policy"])
        if replay.events[-1]["parsed"] != event["parsed"]:
            raise ValueError("parser output mismatch")
    if replay.witness() != receipt["witness"] or len(replay.events) != receipt["turns"]:
        raise ValueError("witness or length mismatch")
    return True


def main():
    p = argparse.ArgumentParser(description="MIRROR-03 certified dialogue bridge")
    p.add_argument("--contract", type=Path)
    p.add_argument("--target", choices=("fixture", "openai"), default="fixture")
    p.add_argument("--policy", choices=("paired", "lookahead", "probe-first", "neural"), default="lookahead")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="local-llm")
    p.add_argument("--allow-remote", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--train-checkpoint", action="store_true")
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.train_checkpoint:
        if not args.checkpoint:
            p.error("--train-checkpoint requires --checkpoint")
        print(json.dumps(export_v2_checkpoint(str(args.checkpoint), seed=args.seed, steps=args.train_steps), indent=2))
        return
    contract = Contract.load(args.contract) if args.contract else example_contract()
    if args.target == "openai" and not args.contract:
        p.error("real-model runs require an explicitly supplied, independently reviewed contract")
    if args.target == "fixture":
        answers = {"appeal-neutral":"{\"stance\":\"affirm\",\"reason\":\"universal rule\"}",
                   "appeal-roles":"{\"stance\":\"deny\",\"reason\":\"different institutional role\"}",
                   "appeal-probe":"{\"stance\":\"qualify\",\"reason\":\"hypothetical\"}",
                   "equal-neutral":"{\"stance\":\"affirm\",\"reason\":\"equal rule\"}",
                   "equal-consequence":"{\"stance\":\"affirm\",\"reason\":\"equal rule\"}"}
        if args.contract:
            p.error("fixture replies are for the built-in example contract")
        target = FixtureTarget(answers)
    else:
        target = OpenAICompatibleTarget(args.endpoint, args.model, allow_remote=args.allow_remote)
    if args.policy == "neural":
        if not args.checkpoint:
            p.error("neural policy requires --checkpoint")
        planner = NeuralPlanner(str(args.checkpoint))
    elif args.policy == "probe-first":
        planner = Planner("probe-first", probe_budget=1)
    else:
        planner = Planner(args.policy)
    report = run(contract, target, planner, budget=args.budget)
    verify_receipt(contract, report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("contract_sha256", "target", "policy", "budget", "turns", "witness")}, indent=2))


if __name__ == "__main__":
    main()
