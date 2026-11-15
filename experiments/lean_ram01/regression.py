"""RAM-01 post-hoc FC-13 theorem regression with live Lean and axiom audit."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import torch
from train import FAMILIES, CONFIGS, SEEDS, digest, load_checkpoint, read_rows

MATHCAN_SHA = 'c84e2025334e9a967c1297425a490f98d5b3d189'
SOURCE_GOALS_SHA = '3df327272bc4768eac2fdfaa60a2cdbead06443e48153781683ee03944568554'
DEADLINE_SECONDS = 20.
MAX_CALLS = 12
ALLOWED_AXIOMS = {'propext', 'Classical.choice', 'Quot.sound'}


def axiom_audit(log: str) -> tuple[bool, list[str]]:
    if log.count('RAM01_AXIOMS_BEGIN') != 1 or log.count('RAM01_AXIOMS_END') != 1:
        return False, []
    section = log.split('RAM01_AXIOMS_BEGIN', 1)[1].split('RAM01_AXIOMS_END', 1)[0]
    if 'does not depend on any axioms' in section:
        return True, []
    match = re.search(r'depends on axioms:\s*\[([^\]]*)\]', section, re.S)
    if not match:
        return False, []
    names = [x.strip() for x in match.group(1).split(',') if x.strip()]
    return set(names) <= ALLOWED_AXIOMS, names


def compile_call(original: str, theorem: str, actions: list[str], probe: bool,
                 timeout: float, prefix: Path) -> dict:
    from research.refutation_bench.fc12 import scratch
    source = scratch.make_source(original, theorem, actions, probe=probe)
    if not probe:
        lines = source.splitlines()
        endings = [i for i, line in enumerate(lines) if re.fullmatch(r'end\s+[\w.]+', line.strip())]
        at = endings[-1] if endings else len(lines)
        lines[at:at] = ['#eval IO.println "RAM01_AXIOMS_BEGIN"',
                        '#print axioms ' + theorem, '#eval IO.println "RAM01_AXIOMS_END"']
        source = '\n'.join(lines)+'\n'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    path = prefix.with_suffix('.lean')
    path.write_text(source, encoding='utf8')
    start = time.perf_counter()
    try:
        p = subprocess.run(['lean', str(path.resolve())], capture_output=True, text=True,
                           timeout=timeout)
        text = p.stdout + p.stderr
        code, timed_out = p.returncode, False
    except subprocess.TimeoutExpired as exc:
        a, b = exc.stdout or b'', exc.stderr or b''
        text = (a.decode('utf8', 'replace') if isinstance(a, bytes) else a)
        text += b.decode('utf8', 'replace') if isinstance(b, bytes) else b
        code, timed_out = 124, True
    wall = time.perf_counter()-start
    prefix.with_suffix('.log').write_text(text, encoding='utf8')
    gate, axioms = (True, []) if probe else axiom_audit(text)
    accepted = code == 0 and not timed_out and gate and 'error:' not in text
    if not probe:
        accepted = accepted and 'sorryAx' not in text and "declaration uses 'sorry'" not in text
    goals = scratch._extract_after_states(text) if accepted and probe else []
    return {'accepted': accepted, 'probe': probe, 'timed_out': timed_out,
            'returncode': code, 'remaining_goals': goals,
            'goal_count': sum(g.count('⊢') for g in goals),
            'axioms': axioms, 'axiom_gate_passed': gate if not probe else None,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'wall_ms': round(wall*1000, 6), 'timeout_seconds': timeout,
            'source_file': str(path), 'log_file': str(prefix.with_suffix('.log'))}


def solve(original: str, theorem: str, state: str, ranker, evidence: Path,
          *, compiler=compile_call, generator=None, clock=time.perf_counter,
          deadline: float = DEADLINE_SECONDS) -> dict:
    if generator is None:
        from research.refutation_bench.fc12 import core
        generator = core.generate
    start = clock()
    end = start + deadline
    chosen, calls = [], []
    accepted = False
    reason = 'transition_budget'
    transitions = 0
    stop = False
    for step in range(3):
        if state.count('⊢') != 1:
            reason = 'multigoal_unsupported'
            break
        if clock() >= end:
            reason = 'hard_deadline'
            break
        pool = generator(state)
        candidates = [(f, s) for f in FAMILIES for s in pool.get(f, [])]
        ordered = ranker.rank(state, candidates) if ranker else candidates
        if Counter(ordered) != Counter(candidates):
            raise AssertionError('policy fabricated or removed a candidate')
        advanced = False
        for family, script in ordered:
            if len(calls) >= MAX_CALLS or clock() >= end:
                reason = 'subprocess_budget' if len(calls) >= MAX_CALLS else 'hard_deadline'
                stop = True
                break
            prefix = chosen+[script]
            result = compiler(original, theorem, prefix, True, min(8., end-clock()),
                              evidence/f'call{len(calls):02d}')
            result.update(family=family, script=script, step=step)
            calls.append(result)
            if clock() >= end:
                reason, stop = 'hard_deadline', True
                break
            if not result['accepted']:
                continue
            if result['goal_count'] > 1:
                reason, stop = 'multigoal_unsupported', True
                break
            if result['goal_count'] == 0:
                if len(calls) >= MAX_CALLS:
                    reason, stop = 'subprocess_budget', True
                    break
                final = compiler(original, theorem, prefix, False, min(8., end-clock()),
                                 evidence/f'call{len(calls):02d}')
                final.update(family=family, script=script, step=step)
                calls.append(final)
                chosen = prefix
                accepted = bool(final['accepted'] and not final['timed_out'] and clock() <= end)
                reason = 'kernel_certified' if accepted else ('hard_deadline' if clock() > end else 'final_rejected')
                stop = True
                break
            following = '\n'.join(result['remaining_goals']).strip()
            if following and following != state.strip():
                state, chosen = following, prefix
                advanced = True
                transitions += 1
                break
        if stop:
            break
        if not advanced:
            reason = 'no_progress'
            break
    elapsed = clock()-start
    if elapsed > deadline:
        accepted = False
        reason = 'hard_deadline'
    return {'accepted': accepted, 'reason': reason, 'chosen_actions': chosen,
            'transitions': transitions, 'lean_subprocesses': len(calls),
            'elapsed_ms': round(elapsed*1000, 6), 'deadline_ms': deadline*1000,
            'calls': calls, 'complete_proof_gate': 'fresh Lean compile plus printed axiom audit'}


def originals(mathcan: Path, output: Path) -> tuple[list[tuple[str, str]], dict, dict]:
    from research.refutation_bench.fc13 import test_data
    if subprocess.check_output(['git', '-C', str(mathcan), 'rev-parse', 'HEAD'], text=True).strip() != MATHCAN_SHA:
        raise ValueError('original MathCan commit differs')
    test_data.emit(mathcan, output)
    for file in test_data.TEST_MODULES:
        stem = Path(file).stem
        proc = subprocess.run(['lean', str((output/(stem+'.lean')).resolve())],
                              capture_output=True, text=True, timeout=120)
        text = proc.stdout+proc.stderr
        (output/(stem+'.log')).write_text(text, encoding='utf8')
        if proc.returncode or 'error:' in text or 'sorryAx' in text:
            raise AssertionError('original proof oracle failed: '+file)
    test_data.extract(output)
    if digest(output/'actual_goal_states.jsonl') != SOURCE_GOALS_SHA:
        raise AssertionError('original compiler goal states changed')
    rows = read_rows(output/'actual_goal_states.jsonl')
    initial = {(r['source_file'], r['theorem']): r['actual_lean_goal_state'] for r in rows if r['stage_index'] == 0}
    cases = [(file, name) for file, names in test_data.TEST_MODULES.items() for name in names]
    if len(cases) != 16 or set(cases) != set(initial):
        raise AssertionError('frozen sixteen-case regression set changed')
    sources = {file: (mathcan/file).read_text(encoding='utf8') for file in test_data.TEST_MODULES}
    return cases, initial, sources


def run(mathcan: Path, weights: Path, out: Path, variant: str) -> dict:
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    cases, states, sources = originals(mathcan, out/'oracle')
    if not (weights/'training_report.json').exists():
        weights = weights/'ram01_trained'
    report = json.loads((weights/'training_report.json').read_text())
    variants = [('native', None)] if variant == 'native' else [(f'{variant}_s{s}', s) for s in SEEDS]
    summaries = []
    for name, seed in variants:
        model = None
        checkpoint_hash = None
        if seed is not None:
            entries = [r for r in report['final'] if r['name'] == variant and r['seed'] == seed]
            if len(entries) != 1:
                raise AssertionError('missing frozen model')
            checkpoint = weights/'final'/entries[0]['checkpoint']
            if digest(checkpoint) != entries[0]['checkpoint_sha256']:
                raise AssertionError('checkpoint hash changed')
            checkpoint_hash = digest(checkpoint)
            model = load_checkpoint(checkpoint)
        results = []
        for case_index, (file, theorem) in enumerate(cases):
            row = solve(sources[file], theorem, states[file, theorem], model,
                        out/'evidence'/name/f'{case_index:02d}')
            row.update(source_file=file, theorem=theorem, model=name, seed=seed)
            results.append(row)
            print('RAM01_PROOF', json.dumps({k: row[k] for k in ('model', 'theorem', 'accepted', 'reason', 'lean_subprocesses', 'elapsed_ms')}), flush=True)
        replays = []
        for i, row in enumerate(results):
            if row['accepted']:
                replay = compile_call(sources[row['source_file']], row['theorem'], row['chosen_actions'],
                                      False, 30., out/'replay'/name/f'{i:02d}')
                if not replay['accepted']:
                    raise AssertionError('reported proof failed fresh replay')
                replays.append(replay)
        path = out/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False)+'\n' for r in results), encoding='utf8')
        n = sum(r['accepted'] for r in results)
        elapsed = sum(r['elapsed_ms'] for r in results)
        summary = {'model': name, 'certified': n, 'cases': 16,
                   'total_wall_ms': elapsed, 'ms_per_proof': elapsed/n if n else None,
                   'subprocesses': sum(r['lean_subprocesses'] for r in results),
                   'reason_counts': dict(Counter(r['reason'] for r in results)),
                   'checkpoint_sha256': checkpoint_hash, 'trajectory_sha256': digest(path),
                   'successful_proofs_recompiled': len(replays)}
        summaries.append(summary)
        print('RAM01_REGRESSION_MODEL', json.dumps(summary, sort_keys=True), flush=True)
    receipt = {'experiment': 'RAM01_posthoc_FC13_regression', 'variant': variant,
               'source_mathcan_sha': MATHCAN_SHA, 'original_oracle_proofs': 16,
               'previously_observed_evaluation_set': True, 'deadline_seconds': DEADLINE_SECONDS,
               'max_lean_subprocesses': MAX_CALLS, 'axioms_allowed': sorted(ALLOWED_AXIOMS),
               'results': summaries}
    (out/'regression_report.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mathcan', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=['native']+[c.name for c in CONFIGS], required=True)
    a = p.parse_args()
    run(a.mathcan, a.weights, a.out, a.variant)
