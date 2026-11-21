"""Index committed continuation reports, including registered unrun amendments.

Presentation only: no model calls, run-file reads, grading, fitting or tests of
scientific hypotheses. Pending published audits remain pending even if a live
worker has advanced. The earlier nine-study index stays a separate snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .evidence_index import read_git, below_normal

ROOT = 'reports/nanbeige_bpd_gym/'
OUTPUTS = (ROOT + 'CONTINUATION_INDEX.md', ROOT + 'continuation_index.json')
SPECS = (
    ('COMP14', 'nan_comp14', 'NAN_COMP14_PROGRESS.md', 'fresh two-world competence; separate first-token/bounded endpoints'),
    ('GRAD15', 'nan_grad15', 'NAN_GRAD15_PROGRESS.md', 'real-checkpoint gradient/resource preflight; zero optimizer steps'),
    ('FIT16', 'nan_fit16', 'NAN_FIT16_PROGRESS.md', 'original sixteen-step fit attempt; partial writer failures preserved'),
    ('XREF17', 'nan_xref17', 'NAN_XREF17_PROGRESS.md', 'original host analysis; unrun under prospective supervision amendment'),
    ('IO18', 'nan_fit16_io18', 'NAN_FIT16_IO18_PROGRESS.md', 'separate execution-only step-log repair; unchanged numerical criteria'),
    ('XREF19', 'nan_xref19', 'NAN_XREF19_PROGRESS.md', 'unchanged SAE analysis bound to passed replacement fit audits'),
    ('PRIME20', 'nan_prime20', 'NAN_PRIME20_PROGRESS.md', 'official local SDK complete one-token reply; same exposed DEV sources'),
    ('CTX21', 'nan_ctx21', 'NAN_CTX21_PROGRESS.md', 'fresh4K/8K source-conditioned graph challenge; explicit capacity/missing coverage'),
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def pick(value, fields):
    return {k: value[k] for k in fields if k in value}


def extract(identifier, data, superseded):
    result = []
    for backend, item in data['backends'].items():
        published = item.get('status', item.get('execution_status', 'unknown'))
        status = published; amendment = None
        if identifier == 'FIT16' and backend == 'bf16':
            amendment = superseded.get('BF16_status')
        elif identifier == 'XREF17':
            amendment = superseded.get('XREF17_status')
        if amendment == 'unrun_registered_supervision_amendment':
            if item.get('cells') or published.startswith('completed'):
                raise ValueError('Conflicting execution and unrun amendment: ' + identifier)
            status = amendment
        row = dict(backend=backend, published_status=published, status=status)
        if amendment:
            row['status_amendment'] = dict(source_report=ROOT + 'nan_fit16_io18_progress.json', supervision_commit=superseded['supervision_commit'])
        if identifier == 'COMP14':
            row.update(pick(item, ('audited_contexts', 'registered_contexts', 'missing_contexts', 'physical_vectors_checked')))
            row['DEV_groups'] = [pick(g, ('arm', 'endpoint', 'correct', 'contexts', 'complete_world_pairs', 'both_world_roots_correct'))
                for g in item['groups'] if g['split'] == 'development']
            for g in row['DEV_groups']:
                if not (0 <= g['correct'] <= g['contexts'] and 0 <= g['both_world_roots_correct'] <= g['complete_world_pairs']):
                    raise ValueError('Invalid competence denominator')
            row['passed_new_screens'] = sum(g['passed'] for g in item['competence_screens'])
        elif identifier == 'GRAD15':
            row['conditions'] = [pick(c, ('passes', 'condition_status', 'successful_gradient_cases', 'archived_cases', 'missing_registered_cases')) for c in item['conditions']]
            row['optimizer_steps'] = data['optimizer_steps']; assert row['optimizer_steps'] == 0
        elif identifier in ('FIT16', 'IO18'):
            cells = {c['passes']: c for c in item['cells']}
            row['cells'] = []
            for passes in (1, 2):
                cell = cells.get(passes)
                if cell is None:
                    row['cells'].append(dict(passes=passes, status=status if amendment else 'awaiting_published_cell_audit', optimizer_steps=None))
                    continue
                value = dict(passes=passes, status=cell['status'], optimizer_steps=cell.get('training', {}).get('optimizer_steps', cell.get('optimizer_steps_completed')))
                value.update(pick(cell, ('factor_update_l2', 'controller_update_l2', 'checkpoint_sha256', 'failure_type')))
                if cell['status'] == 'passed' and value['optimizer_steps'] != 16:
                    raise ValueError('Passed fit lacks all sixteen updates')
                row['cells'].append(value)
        elif identifier in ('XREF17', 'XREF19'):
            cells = {c['passes']: c for c in item['cells']}; audits = {c['passes']: c for c in item.get('arithmetic_audit', [])}
            row['cells'] = []
            for passes in (1, 2):
                c = cells.get(passes)
                if c is None:
                    row['cells'].append(dict(passes=passes, status=status if amendment else 'awaiting_published_host_audit', cached_vectors=None)); continue
                count = sum(d['vectors'] for d in c.get('datasets', []))
                if c['status'] == 'completed':
                    audit = audits.get(passes, {})
                    if audit.get('status') != 'passed' or audit.get('cached_vectors') != count or count != 1706:
                        raise ValueError('Completed cross-reference lacks full independent arithmetic coverage')
                row['cells'].append(dict(passes=passes, status=c['status'], cached_vectors=count if c['status'] == 'completed' else None,
                    physical_geometry_cells=len(c.get('geometry', [])) if c['status'] == 'completed' else None))
        elif identifier == 'CTX21':
            row.update(pick(item, ('audited_contexts','registered_contexts','missing_contexts','physical_vectors_checked')))
            if row['audited_contexts'] + row['missing_contexts'] != row['registered_contexts']:
                raise ValueError('Context challenge audited/missing denominator does not conserve roster')
            row['groups'] = [pick(g, ('split','context_budget','position','arm','loop','correct_worlds','observed_contexts','registered_contexts','missing_contexts',
                'whole_roster_correct_bounds','complete_world_pairs','both_world_roots_correct','whole_roster_both_world_bounds')) for g in item['groups']]
            for g in row['groups']:
                if g['observed_contexts'] + g['missing_contexts'] != g['registered_contexts']:
                    raise ValueError('Context stratum lost missing records')
                if g['observed_contexts'] == 0:
                    if g['correct_worlds'] is not None:
                        raise ValueError('Unobserved context score must remain missing')
                elif g['correct_worlds'] is None or not 0 <= g['correct_worlds'] <= g['observed_contexts']:
                    raise ValueError('Invalid observed context correctness')
            loop2 = [g for g in row['groups'] if g['loop'] == 2]
            if sum(g['observed_contexts'] for g in loop2) != row['audited_contexts'] or sum(g['registered_contexts'] for g in loop2) != row['registered_contexts']:
                raise ValueError('Context stratum denominators do not conserve complete roster')
        else:
            row.update(pick(item, ('registered_contexts', 'successful_official_episodes', 'missing_graded_contexts', 'neural_requests', 'physical_prefix_vectors_byte_exact')))
            if row['successful_official_episodes'] + row['missing_graded_contexts'] != row['registered_contexts']:
                raise ValueError('Official SDK graded/missing coverage does not conserve its roster')
            row['groups'] = [pick(g, ('arm', 'correct', 'graded_contexts', 'missing_graded_contexts', 'complete_world_pairs', 'both_world_roots_correct')) for g in item['groups']]
            for g in row['groups']:
                if not 0 <= g['correct'] <= g['graded_contexts'] or not 0 <= g['both_world_roots_correct'] <= g['complete_world_pairs']:
                    raise ValueError('Invalid SDK correctness or paired-root denominator')
            if sum(g['graded_contexts'] for g in row['groups']) != row['successful_official_episodes']:
                raise ValueError('SDK arm denominators do not equal successful recorded grades')
        result.append(row)
    return result


def build(loader, commit, producer_sha256):
    inputs = {}; seals = {}
    for identifier, folder, markdown, endpoint in SPECS:
        seal_path = f'experiments/{folder}/SOURCE_MANIFEST.json'; report_path = ROOT + folder + '_progress.json'
        raw_seal = loader(seal_path); raw_report = loader(report_path); data = json.loads(raw_report)
        if json.loads(raw_seal).get('sealed_test_accessed') is not False:
            raise ValueError('Source seal changed protected-test assertion: ' + identifier)
        if digest(raw_seal) != data['source_manifest_sha256']:
            raise ValueError('Published source seal mismatch: ' + identifier)
        if data.get('sealed_test_accessed') is not False:
            raise ValueError('Missing protected-test assertion: ' + identifier)
        if data.get('actual_formal_comparisons', 0) != 0 or data.get('planned_formal_comparisons', 0) != 0 or data.get('BPD_donor_transfer_attempted', False):
            raise ValueError('New inference/transfer family requires explicit presentation review')
        inputs[identifier] = data
        seals[identifier] = dict(study=identifier, endpoint=endpoint, source_manifest_path=seal_path,
            source_manifest_sha256=digest(raw_seal), report_path=report_path, report_sha256=digest(raw_report), report_markdown=ROOT + markdown)
    superseded = inputs['IO18'].get('superseded_attempt', {})
    studies = []
    for identifier, *_ in SPECS:
        study = seals[identifier]; study['rows'] = extract(identifier, inputs[identifier], superseded); studies.append(study)
    previous_path = ROOT + 'benchmark_index.json'; previous_raw = loader(previous_path); previous = json.loads(previous_raw)
    result = dict(source_commit=commit, producer_sha256=producer_sha256, studies=studies,
        earlier_nine_study_index=dict(path=previous_path, sha256=digest(previous_raw), source_commit=previous['source_commit'], study_count=len(previous['studies'])),
        superseded_partial_update_audit=superseded.get('partial_update_audit'),
        unit='Stage-specific fresh/reused source roots/families and one checkpoint; no pooling across stages/endpoints/backends.',
        live_process_claim=False, new_model_calls=0, new_grades=0, actual_formal_comparisons=0, BPD_donor_transfer_attempted=False, sealed_test_accessed=False)
    result['evidence_signature'] = digest(json.dumps(result, sort_keys=True).encode())
    return result


def render(data):
    lines = ['# Nanbeige continuation: committed evidence index', '',
        'This indexes published audits and prospectively registered unrun amendments. It reads one immutable Git snapshot, never live runs. A running worker can be ahead of its published audit; pending metrics remain missing. The [earlier nine-study index](BENCHMARK_INDEX.md) is a separate historical snapshot.', '',
        f"Evidence commit: `{data['source_commit']}`. Signature: `{data['evidence_signature']}`.", '',
        '| Study / endpoint | Backend | Published/amended status | Closed evidence |', '|---|---|---|---|']
    for study in data['studies']:
        for row in study['rows']:
            parts = []
            if study['study'] == 'COMP14':
                for arm in ('legacy', 'focused'):
                    g = next(g for g in row['DEV_groups'] if g['arm'] == arm and g['endpoint'] == 'first_token_loop2')
                    parts.append(f"{arm} DEV loop2 first-token {g['correct']}/{g['contexts']}; both-world roots {g['both_world_roots_correct']}/{g['complete_world_pairs']}")
                parts.append(f"{row['audited_contexts']} audited contexts; {row['passed_new_screens']} new screens passed")
            elif study['study'] == 'GRAD15':
                parts = [f"T{c['passes']} {c['condition_status']}: {c['successful_gradient_cases']}/4 gradient cases" for c in row['conditions']]
                parts.append('0 optimizer steps')
            elif 'cells' in row:
                for c in row['cells']:
                    text = f"T{c['passes']} {c['status']}"
                    if c.get('optimizer_steps') is not None:
                        text += f"; {c['optimizer_steps']}/16 updates"
                    if c.get('cached_vectors') is not None:
                        text += f"; {c['cached_vectors']} cached vectors"
                    parts.append(text)
            elif study['study'] == 'CTX21':
                parts = [f"{row['audited_contexts']}/{row['registered_contexts']} audited contexts",f"{row['missing_contexts']} missing outcomes"]
                for g in row['groups']:
                    if g['split']=='development' and g['arm']=='main' and g['loop']==2 and g['observed_contexts']:
                        parts.append(f"{g['context_budget']} {g['position']} main {g['correct_worlds']}/{g['observed_contexts']} observed;{g['missing_contexts']} missing")
            else:
                parts = [f"{row['successful_official_episodes']}/{row['registered_contexts']} successful recorded SDK grades", f"{row['missing_graded_contexts']} missing grades"]
                for g in row['groups']:
                    parts.append(f"{g['arm']} correct {g['correct']}/{g['graded_contexts']}")
            link = study['report_markdown'].split('/')[-1]
            lines.append(f"| [{study['study']}]({link}): {study['endpoint']} | {row['backend']} | {row['status']} | {'; '.join(parts)} |")
    lines += ['', 'The original FIT16 NF4 T1/T2 each stopped after two real updates at the exclusive step-log writer. Independent partial checkpoint counters verified2, while each log contains1 loss and1 missing loss. These partial states are not successful fits. Original BF16 and XREF17 are unrun under the prospective supervision amendment, even though their historical aggregate files still say pending. The IO18 report supplies that explicit status/provenance; no frozen method or old result is relabeled.', '',
        'COMP14 first-token and bounded endpoints, CAL/DEV roots, every copy/erasure control and paired physical recurrence groups remain separate in the full reports. The new source/readout screen does not reopen the failed original algebraic or FIT07 gates. GRAD15 is differentiation/resource evidence with zero training. IO18 is a separate execution-only repair with unchanged numerical criteria and fresh starts, never a partial-checkpoint warm start.', '',
        'XREF19 accepts only passed sixteen-step replacement fit audits, retains every cached vector/SAE feature/control and does not subtract independent backend SAE indices. Prime20 is an official local one-token execution condition on reused DEV sources. Successful recorded SDK grades are not necessarily correct model answers; failed/missing episodes and orphans stay missing, with no response rescue.', '',
        'CTX21 is a separate fresh4K/8K challenge with16 DEV/four CAL roots, early/middle/late graph placement, full-target erasure and literal-copy controls. Capacity/time failures preserve all missing denominators and whole-roster correctness bounds. Pending scores stay missing, not zero. Registration and mock/source checks are not pretrained task results or8K capacity proof.', '',
        'All stages are exposed development work on one checkpoint. Roots, worlds, variants, layers, loops, features and optimizer updates are correlated measurements. No score is pooled across stages, no calibrated causal/sham floor or confirmation power is supplied, and no donor transfer or equivalence is claimed. The original168-case test stays sealed. The continuous goal remains active, with one large model, one numerical CPU thread and Below Normal priority protecting the separate math process.', '',
        '[Exact report/source hashes and whitelisted fields](continuation_index.json) · [Shareable research report](NAN_GYM_RESEARCH_REPORT.md) · [Source-only competence comparison](NAN_COMP14_RESULTS.md) · [Writer amendment](NAN_FIT16_STEP_LOG_AMENDMENT.md)', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source-commit', default='HEAD'); a = p.parse_args(); below_normal()
    commit = subprocess.check_output(['git', 'rev-parse', a.source_commit], text=True).strip()
    producer = 'mechinterp/nanbeige_bpd_gym/campaign_index.py'; raw = read_git(commit, producer)
    assert raw.replace(b'\r\n', b'\n') == Path(producer).read_bytes().replace(b'\r\n', b'\n'), 'Commit the presentation producer first'
    data = build(lambda path: read_git(commit, path), commit, digest(raw))
    Path(OUTPUTS[0]).write_text(render(data), encoding='utf-8', newline='\n')
    Path(OUTPUTS[1]).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(dict(studies=len(data['studies']), evidence_signature=data['evidence_signature'], source_commit=commit))


if __name__ == '__main__':
    main()
