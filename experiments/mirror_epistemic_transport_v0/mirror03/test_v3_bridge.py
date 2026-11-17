"""Offline MIRROR-03 checks; public model calls are never made by this suite."""
from dataclasses import replace
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from experiments.mirror_epistemic_transport_v0.mirror03.contracts import (
    Contract, Certificate, Question, digest, example_contract)
from experiments.mirror_epistemic_transport_v0.mirror03.bridge import (
    Answer, FixtureTarget, Ledger, OpenAICompatibleTarget, Planner,
    parse_answer, run, verify_receipt)


def reply(stance, **extra):
    return json.dumps({"stance":stance, "reason":"explicit test answer", **extra})


def test_compiled_contract_and_fingerprint():
    contract = example_contract()
    assert len(contract.questions) == 5
    assert len(contract.certificates) == 2
    assert Contract.from_dict(contract.canonical_dict()).fingerprint == contract.fingerprint
    assert all(len(c.left_sha256) == 64 for c in contract.certificates)


def test_tampered_question_is_rejected():
    contract = example_contract()
    questions = list(contract.questions)
    questions[0] = replace(questions[0], text="Changed after review")
    with pytest.raises(ValueError, match="changed"):
        replace(contract, questions=tuple(questions)).validate()


def test_rejects_fake_scope_certificate_and_role():
    c = example_contract()
    q0 = c.question_map["appeal-neutral"]
    qprobe = c.question_map["appeal-probe"]
    extra = Certificate(q0.id, qprobe.id, "reviewer", "synthetic example",
                        digest(q0.text), digest(qprobe.text))
    with pytest.raises(ValueError, match="family/scope or contains probes"):
        replace(c, certificates=c.certificates+(extra,)).validate()
    questions = [replace(q, evidence=True) if q.id == "appeal-probe" else q for q in c.questions]
    with pytest.raises(ValueError, match="apparent agreement"):
        replace(c, questions=tuple(questions)).validate()


def test_refuses_duplicate_cells_and_duplicate_question_ids():
    c = example_contract()
    duplicate = replace(c.questions[0], id="other")
    with pytest.raises(ValueError, match="duplicate family/context"):
        replace(c, questions=c.questions + (duplicate,)).validate()
    with pytest.raises(ValueError, match="duplicate question ID"):
        replace(c, questions=c.questions + (c.questions[0],)).validate()


@pytest.mark.parametrize("raw", [
    "yes", (chr(96) * 3) + "json\n{}\n" + (chr(96) * 3), '{"stance":"affirm"}',
    '{"stance":"agree","reason":"ok"}',
    '{"stance":"affirm","reason":"ok","tool":"run"}',
    '{"stance":"deny","reason":"ok","correction":{"question_id":"x","stance":"maybe"}}',
    "[1,2]",
])
def test_fail_closed_on_unstructured_or_malformed_response(raw):
    answer = parse_answer(raw)
    assert not answer.valid
    assert answer.stance == "unclear"


def test_clear_answer_and_qualifier():
    assert parse_answer(reply("affirm")).stance == "affirm"
    assert parse_answer(reply("qualify")).stance == "qualify"
    assert parse_answer(reply("unclear")).stance == "unclear"


def test_verified_witness_requires_two_explicit_answers():
    c = example_contract()
    ledger = Ledger(c)
    ledger.record("appeal-neutral", reply("affirm"))
    ledger.record("appeal-probe", reply("deny"))
    assert ledger.witness() is None
    ledger.record("appeal-roles", reply("qualify"))
    assert ledger.witness() is None
    ledger2 = Ledger(c)
    ledger2.record("appeal-neutral", reply("affirm"))
    ledger2.record("appeal-roles", reply("deny"))
    assert ledger2.witness()["stances"] == {"appeal-neutral":"affirm", "appeal-roles":"deny"}


def test_explicit_correction_supersedes_active_commitment():
    ledger = Ledger(example_contract())
    ledger.record("appeal-neutral", reply("affirm"))
    ledger.record("appeal-roles", reply("deny"))
    assert ledger.witness()
    ledger.record("equal-neutral", reply("affirm", correction={
        "question_id":"appeal-roles", "stance":"affirm"}))
    assert ledger.witness() is None
    assert ledger.active["appeal-roles"].stance == "affirm"


def test_spurious_revision_invalidates_current_reply():
    ledger = Ledger(example_contract())
    ledger.record("appeal-neutral", reply("affirm", correction={
        "question_id":"unasked", "stance":"deny"}))
    assert ledger.active["appeal-neutral"].stance == "unclear"
    assert not ledger.active["appeal-neutral"].valid


def test_target_blind_planners():
    c = example_contract()
    empty = Ledger(c)
    initial = Planner().choose(c, empty)
    assert initial in {"appeal-neutral", "appeal-roles", "equal-neutral", "equal-consequence"}
    assert Planner("probe-first", probe_budget=1).choose(c, empty) == "appeal-probe"
    empty.record("appeal-neutral", reply("affirm"))
    assert Planner().choose(c, empty) == "appeal-roles"
    assert Planner("paired").choose(c, empty) == "appeal-roles"


def test_offline_run_audit_and_tamper_replay():
    c = example_contract()
    t = FixtureTarget({q.id: reply("deny" if q.id == "appeal-roles" else "affirm")
                       for q in c.questions})
    report = run(c, t, Planner("probe-first", probe_budget=1), budget=5)
    assert report["witness"] is not None
    assert len(report["events"]) == report["turns"]
    assert t.history_lengths == list(range(report["turns"]))
    assert verify_receipt(c, report)
    amended = copy.deepcopy(report)
    amended["events"][0]["raw_response"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_receipt(c, amended)
    amended = copy.deepcopy(report)
    amended["witness"] = None
    with pytest.raises(ValueError, match="witness"):
        verify_receipt(c, amended)


def test_remote_target_requires_explicit_https():
    with pytest.raises(ValueError, match="remote"):
        OpenAICompatibleTarget("https://example.com/v1/chat/completions", "model")
    with pytest.raises(ValueError, match="remote"):
        OpenAICompatibleTarget("http://example.com/v1/chat/completions", "model", allow_remote=True)
    with pytest.raises(ValueError, match="invalid endpoint"):
        OpenAICompatibleTarget("http://user:password@127.0.0.1:1/v1/chat/completions", "model")
    assert OpenAICompatibleTarget("https://example.com/v1/chat/completions", "model", allow_remote=True).identity


def test_local_openai_chat_adapter_roundtrip(monkeypatch):
    got = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            got.append(data)
            body = json.dumps({"choices":[{"message":{"content":reply("affirm")}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            return
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        adapter = OpenAICompatibleTarget(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "dummy")
        q = example_contract().questions[0]
        assert parse_answer(adapter.ask(q, [])).stance == "affirm"
        assert parse_answer(adapter.ask(q, [(q.text, reply("deny"))])).stance == "affirm"
        assert len(got) == 2
        assert got[1]["messages"][0]["role"] == "system"
        assert [x["role"] for x in got[1]["messages"]] == ["system", "user", "assistant", "user"]
        assert "api_key" not in json.dumps(got)
    finally:
        server.shutdown()
        worker.join(timeout=2)
        server.server_close()


def test_no_confirmation_from_invalid_target_reply():
    c = example_contract()
    t = FixtureTarget({q.id: "Yes!" for q in c.questions})
    receipt = run(c, t, Planner(), budget=5)
    assert receipt["witness"] is None
    assert verify_receipt(c, receipt)
    assert all(not e["parsed"]["valid"] for e in receipt["events"])
