"""Readable official/native comparison from immutable terminal-audit reports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT='reports/nanbeige_bpd_gym/'
SOURCE=ROOT+'nan_prime20_progress.json'
DEFAULT_COMMIT='76020e2847473e7da54a87b111cec583ac9bcf61'
NATIVE_COMMIT='9db0d4c973fe27662c6b75805d2cabdff9a10bc9'
NATIVE_SOURCE=ROOT+'nan_comp14_progress.json'
SEAL='7df87aeeade4da1fac7064470a6e11afb1a9c209a6119e2afcadf5bea3ce48aa'
NATIVE_SEAL='b9a62f0eedaedc3eca4667412873d5c7151b6a4fb1f98b41bf1bc6978649043f'
ARMS=('focused','copy','focused_erased')
OUTPUTS=(ROOT+'NAN_PRIME20_RESULTS.md',ROOT+'nan_prime20_results_receipt.json')


def native_groups(native,backend):
    result={g['arm']:g for g in native['backends'][backend]['groups'] if g['split']=='development' and g['endpoint']=='first_token_loop2' and g['arm'] in ARMS}
    assert set(result)==set(ARMS)
    assert all(g['contexts']==32 and g['source_roots']==16 and g['complete_world_pairs']==16 for g in result.values())
    return result


def validate(data,native):
    assert data['source_manifest_sha256']==SEAL and native['source_manifest_sha256']==NATIVE_SEAL
    assert native['sealed_test_accessed'] is False
    assert data['sealed_test_accessed'] is False and data['BPD_donor_transfer_attempted'] is False
    assert data['changes_old_competence_gates'] is False and data['original_partial_Prime_coverage_unchanged'] is True
    assert data['planned_formal_comparisons']==data['actual_formal_comparisons']==data['paid_API_calls']==0
    assert set(data['backends'])=={'nf4','bf16'}
    for backend,value in data['backends'].items():
        old=native['backends'][backend]
        assert old['status']=='completed_audited' and old['audited_contexts']==240 and old['physical_vectors_checked']==9600
        reference=native_groups(native,backend);n=value['successful_official_episodes']
        assert value['registered_contexts']==96 and n+value['missing_graded_contexts']==96
        if 'audit_sha256' not in value:
            assert n==0 and not value['groups'] and value['execution_status']!='completed'
            continue
        assert value['status']=='passed' and value['source_manifest_sha256']==SEAL and not value['sealed_test_accessed']
        replies=value['neural_requests'];assert n<=replies<=96
        assert value['matched_native_prefixes']==replies and value['physical_prefix_vectors_byte_exact']==40*replies
        assert value['official_episodes_recorded']==n+value['failed_official_episodes']<=96
        assert value['orphan_or_unbound_neural_replies']==replies-n
        groups=value['groups'];assert len(groups)==3 and {g['arm'] for g in groups}==set(ARMS)
        assert sum(g['graded_contexts'] for g in groups)==n
        for group in groups:
            count=group['graded_contexts'];assert group['registered_contexts']==32 and count+group['missing_graded_contexts']==32
            assert 0<=group['correct']<=group['format_valid']<=count<=32
            assert 0<=group['both_world_roots_correct']<=group['complete_world_pairs']<=16
            lo=group['correct']/32;hi=(group['correct']+32-count)/32
            assert group['whole_registered_correctness_lower_bound']==lo and group['whole_registered_correctness_upper_bound']==hi
            assert lo<=reference[group['arm']]['correct']/32<=hi
        if value['execution_status']=='completed':
            assert n==replies==96 and value['target_unchanged'] and value['failed_official_episodes']==0
            assert all(g['graded_contexts']==32 and g['complete_world_pairs']==16 for g in groups)
            for group in groups:
                old=reference[group['arm']]
                assert group['correct']==old['correct'] and group['both_world_roots_correct']==old['both_world_roots_correct']


def number(v):return 'unknown' if v is None else format(v,'.6g')


def render(data,native,commit,digest):
    validate(data,native)
    lines=['# Official Prime20 and matched native development execution','',
        'Sixteen reused DEV roots, two counterfactual worlds and three arms give96 registered official episodes per backend. Each complete reply contains one unconstrained greedy native loop2 token. Only independently audited official episodes receive grades; pending, missing and orphan replies stay explicit.','',
        '| Backend | Terminal execution status | Neural replies | Recorded official grades / 96 | Missing grades | Failed official episodes | Orphan replies | Exact native prefixes | Byte-exact vectors |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for backend in ('bf16','nf4'):
        r=data['backends'][backend]
        lines.append(f"| {backend} | {r['execution_status']} | {r.get('neural_requests','unknown')} | {r['successful_official_episodes']} / 96 | {r['missing_graded_contexts']} | {r.get('failed_official_episodes','unknown')} | {r.get('orphan_or_unbound_neural_replies','unknown')} | {r.get('matched_native_prefixes','unknown')} | {r.get('physical_prefix_vectors_byte_exact','unknown')} |")
    lines+=['','A successful recorded grade means the episode completed and is auditable; its answer can still be wrong. The vector comparison includes both loops/layers5/11/17/21 at every registered physical site,40 vectors per prefix. This identity is measured against the same backend and initial prompt, without claiming cross-backend quantization or mechanism equivalence.','',
        '| Backend | Arm | Official graded / 32 | Official correct among graded | Missing | Both-correct roots / complete pairs | Whole-roster accuracy bounds | Native first-token correct / 32 |',
        '|---|---|---:|---:|---:|---:|---|---:|']
    for backend in ('bf16','nf4'):
        r=data['backends'][backend];lookup={g['arm']:g for g in r['groups']};reference=native_groups(native,backend)
        for arm in ARMS:
            g=lookup.get(arm);count=g['graded_contexts'] if g else 0
            correct=g['correct'] if count else 'unknown';missing=32-count
            roots=f"{g['both_world_roots_correct']} / {g['complete_world_pairs']}" if g and g['complete_world_pairs'] else 'unknown / 0'
            lo=g['whole_registered_correctness_lower_bound'] if g else 0;hi=g['whole_registered_correctness_upper_bound'] if g else 1
            lines.append(f"| {backend} | {arm} | {count} / 32 | {correct} | {missing} | {roots} | {lo:.5f} to {hi:.5f} | {reference[arm]['correct']} / 32 |")
    lines+=['','Bounds range over missing registered outcomes and are not confidence intervals. Native reference scores remain separately labelled; they never fill an absent official grade. Full coverage requires exact aggregate world and both-world-root agreement with the same-backend COMP14 first-token endpoint. The bounded six-token endpoint and later generation histories are not substituted.','',
        '| Backend | Server elapsed seconds | CUDA peak allocated GiB | CUDA peak reserved GiB | Head device | Target unchanged | SDK outer-whitespace trims |',
        '|---|---:|---:|---:|---|---|---:|']
    for backend in ('bf16','nf4'):
        r=data['backends'][backend];allocated=r.get('peak_cuda_allocated_bytes');reserved=r.get('peak_cuda_reserved_bytes')
        lines.append(f"| {backend} | {number(r.get('elapsed_seconds'))} | {number(allocated/2**30 if allocated is not None else None)} | {number(reserved/2**30 if reserved is not None else None)} | {r.get('head_device','unknown')} | {r.get('target_unchanged','unknown')} | {r.get('SDK_outer_whitespace_trimmed','unknown')} |")
    lines+=['','Server elapsed includes loading, qualification, SDK waits, inference and closing checks; it is not GPU kernel time. Allocator allocated/reserved peaks are distinct from driver residency. NF4 keeps all22 decoder layers and the unchanged BF16 output head on CUDA; BF16 four layers plus a streamed stage and the same head. CPU embeddings, greedy B1, two loops, no KV cache. One numerical CPU thread/Below Normal protects the separate math process.','',
        '## Scope and preserved failures','',
        'This is exposed development execution validation on one checkpoint, not independent task replication or a new competence screen. Roots, worlds, arms, loops, native/SDK runs and backend conditions are correlated. Primary contrast:official complete one-token reply versus same-backend COMP14 initial-token response. Physical neural intervention dose0; scientific causal floor, matched-sham floor and causal power are uncalibrated. Planned/actual formal comparisons0, verdict REDESIGN_OR_PILOT_ONLY. New independent source/readout roots and calibrated causal controls would be needed for confirmation.','',
        'The pinned official local SDK uses subprocess runtime/null harness, one turn/no tools, temperature0/top-p1, no shuffle and max-concurrent1, with no hosted/paid inference. Terminal audit binds task/source IDs, raw assistant nodes, exact prompt IDs, usage/finish, both phase scores, frozen target hashes, full-head parity and every physical vector. The SDK may strip outer whitespace only in its last_reply accessor; the raw node remains bound.','',
        'Original partial PRIME06/PRIME-CTX10 grades remain missing; no lost response, old tail or original competence gate is repaired. There is no SAE fit, online component edit or donor transfer in this condition. Raw replies, tokens, states and per-record grades stay local. Actual process handles and authenticated health supervise live work without ordinary reads of live atomic progress or SDK trace files. Observation expiry never creates a fresh neural attempt. The168-case test stays sealed and the continuous goal remains active.','',
        f"Source commit `{commit}`, report SHA-256 `{digest}`. Native comparator `{NATIVE_COMMIT}`.",'',
        f'[Immutable official aggregate](https://github.com/MoralityLabAI/LoopedTransformers/blob/{commit}/{SOURCE}) · [Live terminal-results stream](NAN_PRIME20_PROGRESS.md) · [Native comparison](NAN_COMP14_RESULTS.md) · [Frozen protocol](../../experiments/nan_prime20/PROTOCOL.md) · [Presentation receipt](nan_prime20_results_receipt.json)','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-commit',default=DEFAULT_COMMIT);a=p.parse_args()
    from .evidence_index import below_normal
    below_normal();commit=subprocess.check_output(['git','rev-parse',a.source_commit],text=True).strip()
    raw=subprocess.check_output(['git','show',commit+':'+SOURCE]);native_raw=subprocess.check_output(['git','show',NATIVE_COMMIT+':'+NATIVE_SOURCE])
    data,native=json.loads(raw),json.loads(native_raw);sha=hashlib.sha256(raw).hexdigest()
    producer='mechinterp/nanbeige_bpd_gym/prime20_results.py';head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    code=subprocess.check_output(['git','show',head+':'+producer]);assert code.replace(b'\r\n',b'\n')==Path(producer).read_bytes().replace(b'\r\n',b'\n')
    Path(OUTPUTS[0]).write_text(render(data,native,commit,sha),encoding='utf-8',newline='\n')
    receipt=dict(source_commit=commit,source_file=SOURCE,source_sha256=sha,native_source_commit=NATIVE_COMMIT,
        native_source_file=NATIVE_SOURCE,native_source_sha256=hashlib.sha256(native_raw).hexdigest(),
        producer_commit=head,producer_sha256=hashlib.sha256(code).hexdigest(),
        terminal_backend_status={b:v['execution_status'] for b,v in data['backends'].items()},
        evidence_class='Host-only presentation of committed terminal audits; no model/grade/fitting or formal test.',sealed_test_accessed=False)
    Path(OUTPUTS[1]).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n');print(receipt)


if __name__=='__main__':main()
