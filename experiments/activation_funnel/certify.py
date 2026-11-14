"""AF-01 independent host-theorem reconstruction and Lean kernel audit.

The learned action is an integer branch selector. This module never executes
model-written Lean. It reconstructs each proof using the fixed original facts
and rule assumptions, checks the fragment independently, and invokes a pinned
Lean executable before recording a kernel certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LEAN_VERSION = "4.19.0"
SCHEMA = "af01-kernel-audit-v1"
ATOM_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


class CertificateError(ValueError):
    """An action or host record cannot support a certificate."""


class KernelAuditError(RuntimeError):
    """The actual compiler or its required axiom receipt failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(episode: Any) -> dict[str, Any]:
    if isinstance(episode, Mapping):
        return dict(episode)
    if hasattr(episode, "to_dict"):
        value = episode.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise CertificateError("episode must be a mapping or expose to_dict()")


def _integer(value: Any, label: str, upper: int | None = None) -> int:
    if type(value) is not int or value < 0 or (upper is not None and value >= upper):
        raise CertificateError(f"invalid {label}")
    return value


def _int_tuple(value: Any, label: str, upper: int) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise CertificateError(f"invalid {label}")
    return tuple(_integer(v, label, upper) for v in value)


def target_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical host statement fields; kept independent of the gym oracle.

    The exact schema is shared with the frozen protocol. Trace selections and
    preview provenance do not add hypotheses to the statement.
    """
    return {
        "schema": "af01-target-v1",
        "atoms": list(data["atoms"]),
        "facts": sorted(data["facts"]),
        "rules": [{"premises": list(r["premises"]), "head": r["head"]}
                  for r in data["rules"]],
        "goal": data["goal"],
        "allowed_axioms": [],
    }


@dataclass(frozen=True)
class HostTheorem:
    case_id: str
    atoms: tuple[str, ...]
    facts: tuple[int, ...]
    rules: tuple[tuple[tuple[int, ...], int], ...]
    goal: int
    candidate_traces: tuple[tuple[int, ...], ...]
    target_digest: str

    @classmethod
    def read(cls, episode: Any) -> "HostTheorem":
        data = _mapping(episode)
        case_id = data.get("case_id")
        if not isinstance(case_id, str) or not case_id or len(case_id) > 256:
            raise CertificateError("invalid case_id")
        raw_atoms = data.get("atoms")
        if not isinstance(raw_atoms, (list, tuple)) or not 1 <= len(raw_atoms) <= 128:
            raise CertificateError("invalid atoms")
        atoms = tuple(raw_atoms)
        if any(not isinstance(a, str) or not ATOM_NAME.fullmatch(a) for a in atoms):
            raise CertificateError("atom names must be bounded plain identifiers")
        if len(set(atoms)) != len(atoms):
            raise CertificateError("duplicate atom names")
        n = len(atoms)
        facts = _int_tuple(data.get("facts"), "facts", n)
        if len(set(facts)) != len(facts):
            raise CertificateError("duplicate facts")
        raw_rules = data.get("rules")
        if not isinstance(raw_rules, (list, tuple)) or not 1 <= len(raw_rules) <= 128:
            raise CertificateError("invalid rules")
        rules = []
        for r in raw_rules:
            if not isinstance(r, Mapping) or set(r) != {"premises", "head"}:
                raise CertificateError("invalid rule fields")
            premises = _int_tuple(r["premises"], "rule premises", n)
            if not premises or len(premises) > 8:
                raise CertificateError("rule must have one to eight premises")
            rules.append((premises, _integer(r["head"], "rule head", n)))
        goal = _integer(data.get("goal"), "goal", n)
        raw_traces = data.get("candidate_traces")
        if not isinstance(raw_traces, (tuple, list)) or len(raw_traces) != 2:
            raise CertificateError("exactly two candidate traces are required")
        traces = tuple(_int_tuple(t, "candidate rule", len(rules)) for t in raw_traces)
        if any(not t or len(t) > 64 for t in traces):
            raise CertificateError("invalid candidate trace length")
        computed = sha256(canonical_bytes(target_payload(data)))
        supplied = data.get("target_digest")
        if not isinstance(supplied, str) or supplied != computed:
            raise CertificateError("host target digest mismatch")
        return cls(case_id, atoms, facts, tuple(rules), goal, traces, computed)


@dataclass(frozen=True)
class ProofCheck:
    accepted: bool
    reason: str
    term: str | None
    rule_count: int


def check_branch(host: HostTheorem, branch: Any) -> ProofCheck:
    """Independent exact check; never consult preview/status/audit labels."""
    branch = _integer(branch, "branch", len(host.candidate_traces))
    known = {fact: f"f{fact}" for fact in host.facts}
    trace = host.candidate_traces[branch]
    for position, rule_id in enumerate(trace):
        premises, head = host.rules[rule_id]
        missing = [p for p in premises if p not in known]
        if missing:
            return ProofCheck(False, f"missing_premise_at_rule_{position}", None, position)
        args = " ".join(f"({known[p]})" for p in premises)
        known[head] = f"r{rule_id} {args}"
    if host.goal not in known:
        return ProofCheck(False, "original_goal_unproved", None, len(trace))
    return ProofCheck(True, "original_goal_derived", known[host.goal], len(trace))


def _candidate_term(host: HostTheorem, branch: int) -> str:
    """Term for native Lean's fixed search, including missing-premise failures.

    Unknown root identifiers are intentionally undefined. Lean must reject that
    alternative; they are never introduced as axioms or theorem parameters.
    """
    known = {fact: f"f{fact}" for fact in host.facts}
    for rule_id in host.candidate_traces[branch]:
        premises, head = host.rules[rule_id]
        args = " ".join(f"({known.get(p, f'af01_unavailable_{p}')})" for p in premises)
        known[head] = f"r{rule_id} {args}"
    return known.get(host.goal, f"af01_unavailable_{host.goal}")


def declaration_name(host: HostTheorem, branch: int | None, prefix: str = "proof") -> str:
    key = canonical_bytes([host.case_id, host.target_digest, branch, prefix])
    return f"af01_{prefix}_{sha256(key)[:24]}"


def _statement(host: HostTheorem, name: str) -> str:
    if not re.fullmatch(r"af01_[a-z]+_[0-9a-f]{24}", name):
        raise CertificateError("invalid generated declaration name")
    props = " ".join(f"P{i}" for i in range(len(host.atoms)))
    binders = [f"({props} : Prop)"]
    binders.extend(f"(f{fact} : P{fact})" for fact in host.facts)
    for i, (premises, head) in enumerate(host.rules):
        rule_type = " → ".join([*(f"P{p}" for p in premises), f"P{head}"])
        binders.append(f"(r{i} : {rule_type})")
    return f"theorem {name} {' '.join(binders)} : P{host.goal} := by\n"


def render_selected(host: HostTheorem, branch: int) -> tuple[str, str]:
    check = check_branch(host, branch)
    if not check.accepted:
        raise CertificateError(check.reason)
    name = declaration_name(host, branch)
    text = _statement(host, name) + f"  exact {check.term}\n\n#print axioms {name}\n"
    return name, text


def render_native_search(host: HostTheorem) -> tuple[str, str]:
    name = declaration_name(host, None, "native")
    body = "  first\n" + "".join(
        f"  | exact {_candidate_term(host, branch)}\n" for branch in (0, 1)
    )
    return name, _statement(host, name) + body + f"\n#print axioms {name}\n"


def check_axiom_output(output: str, names: Sequence[str]) -> None:
    """Require exactly one clean, compiler-generated receipt per declaration."""
    expected = set(names)
    if len(expected) != len(names):
        raise KernelAuditError("duplicate expected declaration")
    clean = re.findall(r"^'([A-Za-z0-9_]+)' does not depend on any axioms\s*$",
                       output, flags=re.MULTILINE)
    if len(clean) != len(expected) or set(clean) != expected:
        raise KernelAuditError("missing, duplicate, or unexpected axiom receipt")
    if re.search(r"depends on axioms|declaration uses 'sorry'|declaration uses `sorry`",
                 output):
        raise KernelAuditError("disallowed proof dependency")


@dataclass(frozen=True)
class LeanCompiler:
    executable: str
    version_output: str
    executable_sha256: str

    @classmethod
    def discover(cls, lean: str = "lean") -> "LeanCompiler":
        resolved = shutil.which(lean)
        if resolved is None:
            raise KernelAuditError(f"required Lean {LEAN_VERSION} executable is unavailable")
        try:
            result = subprocess.run([resolved, "--version"], capture_output=True,
                                    text=True, timeout=15, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise KernelAuditError("cannot inspect Lean executable") from exc
        output = result.stdout + result.stderr
        if result.returncode != 0 or not re.search(r"\bversion 4\.19\.0(?:\b|,)", output):
            raise KernelAuditError(f"required Lean version is {LEAN_VERSION}; observed {output.strip()!r}")
        try:
            digest = sha256(Path(resolved).resolve().read_bytes())
        except OSError as exc:
            raise KernelAuditError("cannot hash Lean executable") from exc
        return cls(str(Path(resolved).resolve()), output.strip(), digest)

    def run(self, source_path: Path, expected_names: Sequence[str],
            timeout: float = 120.0, expect_success: bool = True) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            result = subprocess.run([self.executable, str(source_path.resolve())],
                                    capture_output=True, text=True, timeout=timeout,
                                    check=False, cwd=source_path.parent)
            stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            returncode, timed_out = None, True
        elapsed = time.perf_counter() - start
        log_path = source_path.with_suffix(".lean.log")
        log_path.write_text(stdout + stderr, encoding="utf-8")
        axiom_valid, error = False, None
        if returncode == 0:
            try:
                check_axiom_output(stdout + stderr, expected_names)
                axiom_valid = True
            except KernelAuditError as exc:
                error = str(exc)
        success = returncode == 0 and axiom_valid and not timed_out
        return {
            "source": source_path.name,
            "source_sha256": sha256(source_path.read_bytes()),
            "log": log_path.name,
            "log_sha256": sha256(log_path.read_bytes()),
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed,
            "expected_declarations": list(expected_names),
            "axiom_audit_passed": axiom_valid,
            "kernel_success": success,
            "expect_success": expect_success,
            "expectation_met": success if expect_success else (returncode is not None and returncode != 0),
            "error": error,
        }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compiler_record(compiler: LeanCompiler) -> dict[str, str]:
    return {"version": compiler.version_output,
            "executable_sha256": compiler.executable_sha256}


def certify_selected(episodes: Iterable[Any], rows: Iterable[Mapping[str, Any]],
                     output_dir: str | Path, lean: str = "lean",
                     batch_size: int = 128, timeout: float = 120.0) -> dict[str, Any]:
    """Audit all selected decisions, retaining rejected attempts and stops.

    Rows require case_id, branch (0/1 or None), and target_digest. A missing
    compiler is fatal even when the model selected no valid proofs. The caller
    must separately audit the episode's action/budget trace.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise CertificateError("invalid compiler batch size")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hosts: dict[str, HostTheorem] = {}
    for episode in episodes:
        host = HostTheorem.read(episode)
        if host.case_id in hosts:
            raise CertificateError("duplicate episode case_id")
        hosts[host.case_id] = host
    results, sources = [], {}
    for row_index, original in enumerate(rows):
        row = dict(original)
        case_id = row.get("case_id")
        if case_id not in hosts:
            raise CertificateError("outcome refers to an unknown host episode")
        host = hosts[case_id]
        if row.get("target_digest") != host.target_digest:
            raise CertificateError("outcome target digest mismatch")
        if "branch" not in row:
            raise CertificateError("outcome lacks a branch decision")
        branch = row["branch"]
        record = {"row_index": row_index, "case_id": case_id,
                  "branch": branch, "target_digest": host.target_digest,
                  "arm": row.get("arm"), "seed": row.get("seed"),
                  "condition": row.get("condition"), "kernel_certified": False}
        if branch is None:
            record.update(fragment_accepted=False, decision="stop", reason="stopped")
        else:
            check = check_branch(host, branch)
            record.update(fragment_accepted=check.accepted, decision="submit",
                          reason=check.reason, rule_count=check.rule_count)
            if check.accepted:
                name, source = render_selected(host, branch)
                if name in sources and sources[name] != source:
                    raise CertificateError("generated declaration collision")
                sources[name] = source
                record["declaration"] = name
        results.append(record)
    receipt: dict[str, Any] = {
        "schema": SCHEMA, "required_lean_version": LEAN_VERSION,
        "certifier_source_sha256": sha256(Path(__file__).read_bytes()),
        "scope": "Selected branch proofs of original host theorems; gym trace and budget auditing are separate.",
        "host_count": len(hosts), "decision_count": len(results),
        "fragment_accepted_rows": sum(r["fragment_accepted"] for r in results),
        "rejected_submission_rows": sum(r["decision"] == "submit" and not r["fragment_accepted"] for r in results),
        "stop_rows": sum(r["decision"] == "stop" for r in results),
        "unique_proof_count": len(sources), "compilations": [], "rows": results,
        "kernel_complete": False,
    }
    receipt_path = out / "kernel_audit.json"
    _write_json(receipt_path, receipt)
    try:
        compiler = LeanCompiler.discover(lean)
        receipt["compiler"] = _compiler_record(compiler)
        names = sorted(sources)
        for offset in range(0, len(names), batch_size):
            batch_names = names[offset:offset + batch_size]
            source_path = out / f"Selected_{offset // batch_size:04d}.lean"
            source_path.write_text("\n".join(sources[name] for name in batch_names), encoding="utf-8")
            compiled = compiler.run(source_path, batch_names, timeout)
            receipt["compilations"].append(compiled)
            if not compiled["kernel_success"]:
                raise KernelAuditError(f"Lean proof batch failed: {source_path.name}")
            for record in results:
                if record.get("declaration") in batch_names:
                    record["kernel_certified"] = True
        receipt["kernel_complete"] = True
        receipt["kernel_certified_rows"] = sum(r["kernel_certified"] for r in results)
        receipt["elapsed_lean_seconds"] = sum(r["elapsed_seconds"] for r in receipt["compilations"])
    except KernelAuditError as exc:
        receipt["error"] = str(exc)
        _write_json(receipt_path, receipt)
        raise
    _write_json(receipt_path, receipt)
    return receipt


def native_baseline(episodes: Iterable[Any], output_dir: str | Path,
                    lean: str = "lean", timeout: float = 30.0) -> dict[str, Any]:
    """Frozen lexicographic Lean ``first`` over the two original branch terms.

    Every episode is compiled, including unsolvable contexts. Two alternatives
    and one compiler process are available per episode. Timings are observations;
    no matched-wall-time claim is made.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compiler = LeanCompiler.discover(lean)
    rows = []
    for index, episode in enumerate(episodes):
        host = HostTheorem.read(episode)
        name, source = render_native_search(host)
        path = out / f"Native_{index:05d}.lean"
        path.write_text(source, encoding="utf-8")
        expected_solvable = any(check_branch(host, branch).accepted for branch in (0, 1))
        compiled = compiler.run(path, [name], timeout, expect_success=expected_solvable)
        if not compiled["timed_out"] and compiled["kernel_success"] != expected_solvable:
            raise KernelAuditError(f"native baseline disagrees with independent fragment check: {host.case_id}")
        rows.append({"case_id": host.case_id, "target_digest": host.target_digest,
                     "candidate_order": [0, 1], "candidate_limit": 2,
                     "fragment_solvable": expected_solvable,
                     "compiler_calls": 1, **compiled})
    result = {"schema": "af01-native-lean-baseline-v1",
              "policy": "first exact branch0, then exact branch1",
              "compiler": _compiler_record(compiler), "rows": rows,
              "total": len(rows), "kernel_successes": sum(r["kernel_success"] for r in rows),
              "timeouts": sum(r["timed_out"] for r in rows),
              "elapsed_lean_seconds": sum(r["elapsed_seconds"] for r in rows),
              "timing_scope": "Per-episode compiler process plus native search; measured independently of model work."}
    _write_json(out / "native_baseline.json", result)
    return result


def negative_guards(output_dir: str | Path, lean: str = "lean",
                    timeout: float = 30.0) -> dict[str, Any]:
    """Actual Lean rejection of missing premises and target substitution."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compiler = LeanCompiler.discover(lean)
    positive_name = "af01_proof_000000000000000000000000"
    positive_path = out / "Guard_positive.lean"
    positive_path.write_text(
        f"theorem {positive_name} (P : Prop) (h : P) : P := by\n  exact h\n\n#print axioms {positive_name}\n",
        encoding="utf-8",
    )
    positive = compiler.run(positive_path, [positive_name], timeout)
    cases = {
        "unauthorized_root": "theorem af01_guard_missing (P Q : Prop) (r : P → Q) : Q := by\n  exact r af01_unavailable_root\n",
        "changed_target": "theorem af01_guard_target (P Q : Prop) (h : P) : Q := by\n  exact h\n",
    }
    rows = [{"guard": "positive_control", **positive}]
    for label, source in cases.items():
        path = out / f"Guard_{label}.lean"
        path.write_text(source, encoding="utf-8")
        result = compiler.run(path, [], timeout, expect_success=False)
        rows.append({"guard": label, **result})
    receipt = {"schema": "af01-kernel-negative-guards-v1",
               "compiler": _compiler_record(compiler), "rows": rows,
               "passed": all(r["expectation_met"] for r in rows)}
    _write_json(out / "negative_guards.json", receipt)
    if not receipt["passed"]:
        raise KernelAuditError("Lean failed a negative guard")
    return receipt


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["selected", "baseline", "guards", "all"])
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lean", default="lean")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    episodes = [] if args.episodes is None else read_jsonl(args.episodes)
    if args.command in {"selected", "baseline", "all"} and args.episodes is None:
        parser.error("--episodes is required")
    if args.command in {"selected", "all"} and args.outcomes is None:
        parser.error("--outcomes is required")
    summaries = {}
    if args.command in {"selected", "all"}:
        result = certify_selected(episodes, read_jsonl(args.outcomes), args.out / "selected",
                                  args.lean, args.batch_size, args.timeout)
        summaries["selected"] = {k: v for k, v in result.items() if k not in {"rows", "compilations"}}
    if args.command in {"baseline", "all"}:
        result = native_baseline(episodes, args.out / "baseline", args.lean, args.timeout)
        summaries["baseline"] = {k: v for k, v in result.items() if k != "rows"}
    if args.command in {"guards", "all"}:
        summaries["guards"] = negative_guards(args.out / "guards", args.lean, args.timeout)
    args.out.mkdir(parents=True, exist_ok=True)
    summaries["inputs"] = {
        "episodes_sha256": sha256(args.episodes.read_bytes()) if args.episodes else None,
        "outcomes_sha256": sha256(args.outcomes.read_bytes()) if args.outcomes else None,
    }
    _write_json(args.out / "summary.json", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
