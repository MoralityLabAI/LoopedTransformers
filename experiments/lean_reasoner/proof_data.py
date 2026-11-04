"""LR-01: synthetic proof terms for a finite, acyclic Horn fragment of Lean.

This module is dependency-free. All propositions and implications are explicit
hypotheses. The Python checker covers this fragment only. Lean acceptance is
recorded separately; generating Lean source does not imply kernel acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def structure_id(t):
    """Isomorphism-invariant WL fingerprint; collisions conservatively group tasks."""
    n = t['n']; facts = set(t['facts']); goal = t['goal']
    colors = [digest([i in facts, i == goal]) for i in range(n)]
    for _ in range(n):
        new = []
        for i in range(n):
            parents = [sorted(colors[p] for p in ps) for ps, h in t['rules'] if h == i]
            children = sorted([colors[h], sorted(colors[p] for p in ps if p != i)]
                              for ps, h in t['rules'] if i in ps)
            new.append(digest([colors[i], parents, children]))
        colors = new
    return digest(sorted(colors))


def required(t, known=None):
    """Exact dependency oracle for the unique-producer fragment; never a feature."""
    known = set(t['facts']) if known is None else set(known)
    producer = {h: i for i, (_, h) in enumerate(t['rules'])}
    out = set(); visiting = set()
    def visit(p):
        if p in known:
            return
        if p in visiting or p not in producer:
            raise ValueError('cyclic or unprovable obligation')
        visiting.add(p); i = producer[p]
        for q in t['rules'][i][0]:
            visit(q)
        visiting.remove(p); out.add(i)
    visit(t['goal'])
    return out


def legal(t, known):
    return [i for i, (ps, h) in enumerate(t['rules']) if h not in known and set(ps) <= known]


def proof_term(t):
    producers = {h: i for i, (_, h) in enumerate(t['rules'])}
    def go(p):
        if p in t['facts']:
            return ['hyp', p]
        i = producers[p]
        return ['app', i, [go(q) for q in t['rules'][i][0]]]
    return go(t['goal'])


def check_term(t, term):
    """Independent recursive type checker, without dependency-oracle calls."""
    if not isinstance(term, list) or not term:
        raise ValueError('malformed term')
    if term[0] == 'hyp' and len(term) == 2:
        if term[1] not in t['facts']:
            raise ValueError('unknown hypothesis')
        return term[1]
    if term[0] == 'app' and len(term) == 3:
        i, args = term[1:]
        if not isinstance(i, int) or not 0 <= i < len(t['rules']):
            raise ValueError('unknown rule')
        ps, h = t['rules'][i]
        if len(args) != len(ps) or [check_term(t, a) for a in args] != ps:
            raise ValueError('argument type mismatch')
        return h
    raise ValueError('unknown constructor')


def trace_from_term(term):
    """Postorder application trace extracted from the proof AST, sharing DAG nodes."""
    seen = set(); trace = []
    def go(x):
        if x[0] == 'app':
            for a in x[2]: go(a)
            if x[1] not in seen:
                seen.add(x[1]); trace.append(x[1])
    go(term)
    return trace


def check_trace(t, trace):
    """Replay proposals; construct a typed term from accepted applications."""
    terms = {p: ['hyp', p] for p in t['facts']}
    for i in trace:
        if i not in legal(t, set(terms)):
            raise ValueError('illegal application')
        ps, h = t['rules'][i]
        terms[h] = ['app', i, [terms[p] for p in ps]]
    if t['goal'] not in terms:
        raise ValueError('unfinished proof')
    term = terms[t['goal']]
    if check_term(t, term) != t['goal']:
        raise ValueError('wrong conclusion')
    return term


def lean_theorem(t, name, trace=None):
    trace = trace_from_term(t['term']) if trace is None else trace
    check_trace(t, trace)
    params = ['(' + ' '.join(f'P{i}' for i in range(t['n'])) + ' : Prop)']
    params += [f'(h{p} : P{p})' for p in sorted(t['facts'])]
    params += [f'(r{i} : ' + ' → '.join(f'P{p}' for p in ps + [h]) + ')'
               for i, (ps, h) in enumerate(t['rules'])]
    lines = [f'theorem {name} ' + ' '.join(params) + f' : P{t["goal"]} := by']
    names = {p: f'h{p}' for p in t['facts']}
    for k, i in enumerate(trace):
        ps, h = t['rules'][i]
        lines.append(f'  have s{k} : P{h} := r{i} ' + ' '.join(names[p] for p in ps))
        names[h] = f's{k}'
    lines.append('  exact ' + names[t['goal']])
    return '\n'.join(lines)


def generate_one(rng, deep=False):
    n = rng.randint(18, 23) if deep else rng.randint(11, 17)
    f = rng.randint(2, 4); rules = []
    for h in range(f, n):
        k = 2 if rng.random() < .48 else 1
        ps = rng.sample(range(h), k)
        rules.append([ps, h])
    goal = rng.randrange(max(f + 3, n // 2), n)
    t = {'n': n, 'facts': list(range(f)), 'rules': rules, 'goal': goal}
    depth = [0] * n
    for ps, h in rules: depth[h] = 1 + max(depth[p] for p in ps)
    need = required(t)
    if not ((5 <= depth[goal] <= 7 and 6 <= len(need) <= 12) if deep
            else (2 <= depth[goal] <= 4 and 2 <= len(need) <= 7)):
        return None
    # Relabel atoms and shuffle rule positions before any feature extraction.
    perm = list(range(n)); rng.shuffle(perm); rng.shuffle(rules)
    t = {'n': n, 'facts': sorted(perm[p] for p in range(f)),
         'rules': [[[perm[p] for p in ps], perm[h]] for ps, h in rules],
         'goal': perm[goal], 'depth': depth[goal], 'min_steps': len(need)}
    t['structure_id'] = structure_id(t)
    t['term'] = proof_term(t)
    assert check_term(t, t['term']) == t['goal']
    t['trace'] = trace_from_term(t['term'])
    check_trace(t, t['trace'])
    return t


def build(out, counts=(768, 128, 192, 192), seed=20261105):
    out.mkdir(parents=True, exist_ok=True); rng = random.Random(seed)
    used = set(); tasks = []; attempts = 0
    for split, count in zip(['train', 'val', 'test', 'depth_ood'], counts):
        kept = 0
        while kept < count:
            attempts += 1
            if attempts > 200000: raise RuntimeError('generation exhausted')
            t = generate_one(rng, split == 'depth_ood')
            if t is None or t['structure_id'] in used: continue
            used.add(t['structure_id']); t['split'] = split
            t['id'] = f'{split}_{kept:04d}'; tasks.append(t); kept += 1
    path = out / 'tasks.jsonl'
    path.write_text(''.join(json.dumps(t, sort_keys=True) + '\n' for t in tasks))
    lean = 'set_option linter.unusedVariables false\n\n' + '\n\n'.join(
        lean_theorem(t, t['id']) for t in tasks) + '\n'
    (out / 'Corpus.lean').write_text(lean)
    manifest = {'schema': 'LR-01/v1', 'seed': seed, 'counts': dict(zip(
        ['train','val','test','depth_ood'], counts)), 'attempts': attempts,
        'unique_structure_groups': len(used), 'tasks_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'lean_source_sha256': hashlib.sha256(lean.encode()).hexdigest(),
        'python_term_checks': len(tasks), 'python_trace_checks': len(tasks),
        'lean_kernel_status': 'not_run', 'fragment': 'acyclic unique-producer propositional Horn'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return tasks, manifest


def verify_lean(out, executable='lean'):
    path = shutil.which(executable)
    if path is None:
        result = {'status': 'unavailable', 'reason': 'Lean executable absent'}
    else:
        version = subprocess.check_output([path, '--version'], text=True).strip()
        files = [out / 'Corpus.lean'] + sorted(out.glob('Predictions*.lean'))
        runs = []
        for source in files:
            p = subprocess.run([path, str(source.resolve())], capture_output=True, text=True, timeout=180)
            runs.append({'file': source.name, 'returncode': p.returncode,
                         'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'stdout': p.stdout, 'stderr': p.stderr})
        result = {'status': 'passed' if all(x['returncode'] == 0 for x in runs) else 'failed',
                  'version': version, 'files': runs}
    (out / 'lean_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['generate','verify'])
    p.add_argument('--out', type=Path, default=Path(__file__).parent / 'artifacts')
    p.add_argument('--lean', default='lean'); a = p.parse_args()
    result = build(a.out)[1] if a.command == 'generate' else verify_lean(a.out, a.lean)
    print(json.dumps(result, indent=2))
    if a.command == 'verify' and result['status'] != 'passed': raise SystemExit(2)
