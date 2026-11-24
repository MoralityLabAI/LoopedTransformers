"""Public terminal-audit-only SDK accuracy and exact-prefix coverage."""
from datetime import datetime, timezone
import json
from pathlib import Path
from .common import ROOT, RUNS, read, rows, sha, verify

OUT = Path('reports/nanbeige_bpd_gym')
OUTPUTS = [OUT / 'NAN_PRIME20_PROGRESS.md', OUT / 'nan_prime20_progress.json']


def report():
    verify(); backends = {}; events = rows(RUNS / 'queue_events.jsonl') if (RUNS / 'queue_events.jsonl').exists() else []
    for backend in ('nf4', 'bf16'):
        value = dict(execution_status='pending', registered_contexts=96, successful_official_episodes=0, missing_graded_contexts=96, groups=[])
        failures = [e for e in events if e.get('backend') == backend and e['event'] in ('audit_terminal', 'model_worker_terminal') and e['exit_code']]
        if failures:
            value.update(execution_status='failed_attempt_or_audit_preserved', failures=failures)
        path = RUNS / 'audits' / (backend + '.json')
        if path.exists():
            audit = read(path); assert audit['status'] == 'passed' and audit['source_manifest_sha256'] == sha(ROOT / 'SOURCE_MANIFEST.json')
            terminal = RUNS / backend / 'receipt.json'; assert sha(terminal) == audit['worker_receipt_sha256']
            value.update(audit, audit_sha256=sha(path)); receipt = read(terminal)
            for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'elapsed_seconds', 'head_device', 'target_unchanged'):
                value[key] = receipt.get(key)
        backends[backend] = value
    payload = dict(experiment='NAN-PRIME-20', updated_UTC=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), backends=backends,
        evidence_class='Official local SDK execution condition on exposed COMP14 DEV sources; no independent replication.',
        unit='Sixteen reused DEV roots; worlds, arms, loops, SDK/native runs and backend checkpoint conditions correlated.',
        primary_contrast='Complete official one-token final reply versus same-backend COMP14 loop2 initial-token endpoint.',
        physical_dose=0, scientific_causal_floor=None, matched_sham_floor=None, estimated_causal_power='Not calibrated; execution audit only.',
        readout_selection_exposure='COMP14 outcomes/GRAD15 and original failed fitting exposed; current fitting separate and not used for source selection.',
        planned_formal_comparisons=0, actual_formal_comparisons=0, paid_API_calls=0, BPD_donor_transfer_attempted=False,
        changes_old_competence_gates=False, original_partial_Prime_coverage_unchanged=True, sealed_test_accessed=False)
    OUT.mkdir(exist_ok=True)
    OUTPUTS[1].write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    lines = ['# NAN-PRIME-20: official local competence SDK execution', '',
        'Exact COMP14 DEV source order:16 roots ×2 counterfactual worlds ×focused/copy/focused-erased arms =96 official episodes/backend. One unconstrained native loop2 token per complete reply, matching the separately recorded COMP14 first-token endpoint. This is exposed execution validation, not new source competence, an old-tail rescue or donor transfer.', '',
        '| Backend | Execution status | Fresh replies | Successful official grades | Missing grades | Native prefixes matched | Byte-exact physical vectors |',
        '|---|---|---:|---:|---:|---:|---:|']
    for b, v in backends.items():
        lines.append(f"| {b} | {v['execution_status']} | {v.get('neural_requests', '—')} | {v['successful_official_episodes']}/96 | {v['missing_graded_contexts']} | {v.get('matched_native_prefixes', '—')} | {v.get('physical_prefix_vectors_byte_exact', '—')} |")
    lines += ['', 'Only independent terminal audits produce grades. Missing/failed episodes remain missing even if a local neural reply exists. Bounds below range over the registered roster; they are missing-coverage bounds, not confidence intervals.', '',
        '| Backend | Arm | Correct worlds | Graded worlds | Both-correct roots / complete pairs | Whole-roster lower–upper accuracy |', '|---|---|---:|---:|---:|---:|']
    for b, v in backends.items():
        for g in v['groups']:
            lines.append(f"| {b} | {g['arm']} | {g['correct']} | {g['graded_contexts']}/32 | {g['both_world_roots_correct']}/{g['complete_world_pairs']} | {g['whole_registered_correctness_lower_bound']:.4f}–{g['whole_registered_correctness_upper_bound']:.4f} |")
    lines += ['', 'Distinct source IDs preserve legitimate duplicate erased prompts. Source slots follow exact no-shuffle/one-concurrent SDK order, with no adjacent identical prompts. In-flight and immediate replay attempts cannot create another neural response. Each successful episode must bind its source ID, fresh request, exact raw assistant node, usage and finish reason. SDK last_reply may strip outer whitespace; raw text stays bound. No cached reply is served.', '',
        'Both native phase scores, prompt IDs and all40 physical vectors at layers5/11/17/21 must match that backend’s original COMP14 prefix. The full-vocabulary parity check uses one fixed CAL prefix before serving, outside the96 official DEV tasks. Such byte identity is an execution measurement, not quantization or mechanism equivalence. Later generation histories and bounded-six-token scores are not substituted.', '',
        'The pinned official SDK ed70053 uses persistent local Linux Python, subprocess runtime/null harness, one turn, temperature0/top-p1 and no hosted or paid inference. NF4 retains22 decoder layers and unchanged BF16 head on GPU; BF16 retains four layers/stage and that head. CPU embeddings preserve capacity. One numerical CPU thread/Below Normal and serialized audits protect the separate math process. Allocator peaks are not driver residency certificates.', '',
        'The queue observes authenticated in-memory health and actual process handles, never live atomic progress or SDK trace files. Both model and SDK terminate before audits. Errors, orphan replies, failed/missing episodes and observation overruns remain explicit; no automatic fresh neural retry. Only the exact owned SDK process may be stopped after verified model failure. The original168-case test stays sealed, prior partial Prime studies remain unchanged, and the continuous goal remains active.', '',
        'All source fixtures are public synthetic development data. Generated traces, replies, tokens, vectors and per-record grades remain local; no SAE refitting, feature-index subtraction, online learned-factor edit or donor intervention occurs here.', '',
        f"Source seal: `{payload['source_manifest_sha256']}`.", '',
        '[Frozen protocol](../../experiments/nan_prime20/PROTOCOL.md) · [Source seal](../../experiments/nan_prime20/SOURCE_MANIFEST.json) · [Full aggregates](nan_prime20_progress.json)', '']
    OUTPUTS[0].write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print({b: v['execution_status'] for b, v in backends.items()}, flush=True)


if __name__ == '__main__':
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); report()
