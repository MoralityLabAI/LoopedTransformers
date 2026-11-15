"""RAM-09 stage 0: bounded retrospective closure witnesses from RAM-08 Lean effects.

A *witness* is a path through independently replayed compiler outcomes, not
an independently certified complete theorem. Unseen successors are censored.
This module never uses prospective PEF theorem source, labels or checkpoints.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import zipfile

ARCHIVE_SHA = 'a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc'
TRANSITIONS_SHA = 'f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f'
Key = tuple[str, str, str]

def goal_signature(goals: list[str]) -> str:
    """Byte-for-byte RAM-08 state-signature canonicalization."""
    mapping: dict[str, str] = {}
    def variable(match):
        symbol = match.group()
        return mapping.setdefault(symbol, f'?m{len(mapping)}')
    out = []
    for goal in goals:
        if goal.count('⊢') != 1:
            raise ValueError('observed Lean goal has an invalid turnstile count')
        lines = [x.strip() for x in goal.splitlines()
                 if not re.match(r'^\s*case\s+', x)]
        canonical = '\n'.join(' '.join(x.split()) for x in lines if x)
        out.append(re.sub(r'\?m(?:\.[A-Za-z_]+)*\.\d+', variable, canonical))
    data = json.dumps(out, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()

def load_archive(path: Path) -> tuple[list[dict], dict]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('RAM-08 archived training ZIP SHA mismatch')
    with zipfile.ZipFile(path) as archive:
        raw = archive.read('data/transitions.jsonl')
        if hashlib.sha256(raw).hexdigest() != TRANSITIONS_SHA:
            raise ValueError('RAM-08 transitions SHA mismatch')
        audit = json.loads(archive.read('data/independent_audit.json'))
        if (audit['status'], audit['checked'], audit['dataset_sha256']) != ('passed', 20930, TRANSITIONS_SHA):
            raise ValueError('RAM-08 independent Lean replay receipt mismatch')
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        if len(rows) != 20930:
            raise ValueError('missing original compiler-observation rows')
        return rows, audit

def key(row: dict) -> Key:
    return (row['source_file'], row['theorem'], row['state_sha256'])

class ObservedClosureGraph:
    """Per-root graph of actual recorded compiler-accepted tactic transitions.

    A shortest witness gives an *upper* bound on the optimal action count over
    every real compiler-supported path, assuming the stored effects replay.
    Absence of a witness has no converse implication: unobserved successors
    may close. A path is not a kernel certificate for the full theorem.
    """
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.by_state: dict[Key, list[dict]] = defaultdict(list)
        self.forward: dict[Key, list[tuple[Key | None, str]]] = defaultdict(list)
        self.reverse: dict[Key, list[tuple[Key, str]]] = defaultdict(list)
        self.distance: dict[Key, int] = {}
        self.best: dict[Key, tuple[Key | None, str]] = {}
        self.censored = 0
        self.matched = 0
        self.terminals = 0
        self.invalid_or_unchanged = 0
        self.theorems = set()
        self._construct()

    def _construct(self):
        for row in self.rows:
            start = key(row)
            self.theorems.add(start[:2])
            if goal_signature(row['before']['goals']) != row['state_sha256']:
                raise ValueError('RAM-08 observation has a changed original before-state SHA')
            if row['label'] not in (0, 1, 2, 3):
                raise ValueError('unknown verified tactic-effect label')
            self.by_state[start].append(row)
        for start, original_rows in self.by_state.items():
            for row in original_rows:
                after = row['after']
                if row['label'] == 0:
                    self.invalid_or_unchanged += 1
                    continue
                if not after.get('accepted'):
                    raise ValueError('positive effect lacks original Lean acceptance')
                if row['label'] == 3:
                    if after['goals']:
                        raise ValueError('registered goal-close label retains open goals')
                    self.terminals += 1
                    self.forward[start].append((None, row['script']))
                    continue
                if not after['goals']:
                    raise ValueError('registered nonclosure label has no pending goals')
                nxt = (start[0], start[1], goal_signature(after['goals']))
                if nxt == start:
                    raise ValueError('positive effect returns to identical original goal state')
                if nxt not in self.by_state:
                    self.censored += 1
                    continue
                self.matched += 1
                self.forward[start].append((nxt, row['script']))
                self.reverse[nxt].append((start, row['script']))
        agenda = deque()
        for start, edges in self.forward.items():
            for successor, script in edges:
                if successor is None and start not in self.distance:
                    self.distance[start] = 1
                    self.best[start] = (None, script)
                    agenda.append(start)
        while agenda:
            successor = agenda.popleft()
            for predecessor, script in self.reverse.get(successor, ()):
                if predecessor not in self.distance:
                    self.distance[predecessor] = 1 + self.distance[successor]
                    self.best[predecessor] = (successor, script)
                    agenda.append(predecessor)

    def witness(self, start: Key) -> list[str] | None:
        if start not in self.distance:
            return None
        chain = []
        current = start
        while current is not None:
            successor, script = self.best[current]
            chain.append(script)
            current = successor
        if len(chain) != self.distance[start]:
            raise AssertionError('shortest witness reconstruction mismatch')
        return chain

    def action_distance(self, row: dict) -> int | None:
        """Actions remaining to closure if this action is selected now.

        Only compiler-verified closing actions or successors in the observed
        finite graph have known values. Unknown progress is *censored*.
        """
        if row['label'] == 0:
            return None
        if row['label'] == 3:
            return 1
        k = key(row)
        end = (k[0], k[1], goal_signature(row['after']['goals']))
        if end not in self.distance:
            return None
        return 1 + self.distance[end]

    def summary(self) -> dict:
        roots = Counter((k[0], k[1]) for k in self.distance)
        return dict(theorems=len(self.theorems), states=len(self.by_state),
                    compiler_observations=len(self.rows), goal_closing_actions=self.terminals,
                    matched_nonterminal_edges=self.matched,
                    unobserved_nonterminal_successors=self.censored,
                    failed_or_unchanged=self.invalid_or_unchanged,
                    states_with_observed_closure_path=len(self.distance),
                    shortest_witness_lengths=dict(sorted(Counter(self.distance.values()).items())),
                    theorem_roots_with_observed_witness=len(roots),
                    unknown_is_censored=True, full_theorem_lean_certificates=0)

def source_splits(rows):
    train = [r for r in rows if not r['development']]
    dev = [r for r in rows if r['development']]
    train_roots = {(r['source_file'], r['theorem']) for r in train}
    dev_roots = {(r['source_file'], r['theorem']) for r in dev}
    if train_roots & dev_roots:
        raise ValueError('original MathCan theorem root is in TRAIN and DEV')
    if {r['state_sha256'] for r in train} & {r['state_sha256'] for r in dev}:
        raise ValueError('original Lean goal hashes overlap TRAIN and DEV')
    return train, dev

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('output path already exists; preserve previous audit')
    rows, _ = load_archive(a.archive)
    train, dev = source_splits(rows)
    tr = ObservedClosureGraph(train)
    dv = ObservedClosureGraph(dev)
    result = dict(campaign='RAM-09/offline_stage0_retrospective',
                  ram08_frozen_archive_sha256=ARCHIVE_SHA,
                  original_compiler_labels_sha256=TRANSITIONS_SHA,
                  prospective_pef_theorems_read=0,
                  prospective_pef_proof_scores=0,
                  source_train_sha='c84e2025334e9a967c1297425a490f98d5b3d189',
                  trained_or_refitted_models=0,
                  graph_train=tr.summary(), graph_old_development=dv.summary(),
                  model_claim='none; graph reachability only',
                  caveat='Compiler-observed tactic paths are not whole-theorem Lean certificates; lack of observed path is censored.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
