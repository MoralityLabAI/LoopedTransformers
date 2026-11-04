"""Train a tied, typed graph-attention proof policy on proof-AST dependencies.

Learned: priority of rule application. Exact: incidence, applicability, term
assembly and fragment checking. There is no language decoder or target LLM.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import resource
import time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from proof_data import check_trace, legal, lean_theorem, trace_from_term

MAX_N = 24
MAX_R = 24


def encode(t, known):
    a = torch.zeros(MAX_N, 3); r = torch.zeros(MAX_R, 4)
    prem = torch.zeros(MAX_R, MAX_N); head = torch.zeros(MAX_R, MAX_N)
    real = torch.zeros(MAX_R, dtype=torch.bool)
    for p in range(t['n']): a[p] = torch.tensor([float(p in known), float(p == t['goal']), 1.])
    for i, (ps, h) in enumerate(t['rules']):
        real[i] = True; prem[i, ps] = 1.; head[i, h] = 1.
        r[i] = torch.tensor([len(ps) / 2., float(h == t['goal']),
                            float(set(ps) <= known and h not in known), float(h in known)])
    return a, r, prem, head, real


def stack(states):
    return tuple(torch.stack([x[i] for x in states]) for i in range(5))


def examples(tasks):
    states = []; labels = []
    for t in tasks:
        known = set(t['facts']); trace = trace_from_term(t['term'])
        for j, action in enumerate(trace):
            states.append(encode(t, known))
            y = torch.zeros(MAX_R); y[trace[j:]] = 1.; labels.append(y)
            known.add(t['rules'][action][1])
    return stack(states), torch.stack(labels)


class TypedLoop(nn.Module):
    def __init__(self, width=32):
        super().__init__(); d = width
        self.rule_update = nn.Sequential(nn.Linear(3*d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.atom_update = nn.Sequential(nn.Linear(3*d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.na = nn.LayerNorm(d); self.nr = nn.LayerNorm(d)

    def attention(self, a, r, incidence):
        # Empty neighborhoods return zero. Only declared rule/atom incidences mix.
        s = self.q(a) @ self.k(r).transpose(1, 2) / math.sqrt(a.shape[-1])
        mask = incidence.transpose(1, 2) > 0
        w = torch.softmax(s.masked_fill(~mask, -1e4), -1) * mask
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        return w @ self.v(r)

    def forward(self, a, r, prem, head):
        parents = prem @ a / prem.sum(-1, keepdim=True).clamp_min(1.)
        r = self.nr(r + self.rule_update(torch.cat([r, head @ a, parents], -1)))
        incoming = self.attention(a, r, head)
        outgoing = self.attention(a, r, prem)
        a = self.na(a + self.atom_update(torch.cat([a, incoming, outgoing], -1)))
        return a, r


class ProofPolicy(nn.Module):
    def __init__(self, loops=4, width=32):
        super().__init__(); self.loops = loops
        self.atom = nn.Linear(3, width); self.rule = nn.Linear(4, width)
        self.cell = TypedLoop(width); self.score = nn.Linear(3*width, 1)

    def forward(self, batch, loops=None, remove_goal=False):
        af, rf, prem, head, real = batch
        if remove_goal:
            af = af.clone(); rf = rf.clone(); af[:,:,1] = 0; rf[:,:,1] = 0
        a = self.atom(af); r = self.rule(rf)
        for _ in range(self.loops if loops is None else loops):
            a, r = self.cell(a, r, prem, head)
        parents = prem @ a / prem.sum(-1, keepdim=True).clamp_min(1.)
        features = torch.cat([r, head @ a, parents], -1)
        return self.score(features).squeeze(-1).masked_fill(~real, -1e4)


def objective(logits, batch, y):
    _, rf, _, _, real = batch
    outstanding = real & (rf[:,:,3] < .5)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    bce = (bce * outstanding).sum() / outstanding.sum().clamp_min(1)
    allowed = real & (rf[:,:,2] > .5)
    positive = allowed & (y > .5)
    choice = (torch.logsumexp(logits.masked_fill(~allowed, -1e4), -1)
              - torch.logsumexp(logits.masked_fill(~positive, -1e4), -1)).mean()
    return bce + choice


@torch.no_grad()
def evaluate(model, tasks, loops=None, remove_goal=False, baseline=None, seed=17):
    rng = random.Random(seed); known = [set(t['facts']) for t in tasks]
    traces = [[] for _ in tasks]
    budgets = [t['min_steps'] for t in tasks]
    for step in range(max(budgets)):
        active = [j for j, t in enumerate(tasks) if step < budgets[j] and t['goal'] not in known[j]]
        if not active: break
        logits = None
        if model is not None:
            batch = stack([encode(tasks[j], known[j]) for j in active])
            logits = model(batch, loops, remove_goal)
            allowed = batch[1][:,:,2] > .5
            logits = logits.masked_fill(~allowed, -1e4)
        for row, j in enumerate(active):
            t = tasks[j]; candidates = legal(t, known[j])
            if not candidates: continue
            if model is not None:
                action = int(logits[row].argmax())
            elif baseline == 'oracle':
                remaining = set(trace_from_term(t['term']))
                action = rng.choice([i for i in candidates if i in remaining])
            elif baseline == 'direct_goal':
                direct = [i for i in candidates if t['rules'][i][1] == t['goal']]
                action = rng.choice(direct or candidates)
            else:
                action = rng.choice(candidates)
            assert action in candidates
            traces[j].append(action); known[j].add(t['rules'][action][1])
    rows = []
    for t, k, trace in zip(tasks, known, traces):
        solved = t['goal'] in k
        if solved: check_trace(t, trace)
        rows.append({'id': t['id'], 'solved': solved, 'steps': len(trace),
                     'budget': t['min_steps'], 'trace': trace})
    return {'solved': sum(r['solved'] for r in rows), 'total': len(rows),
            'rate': sum(r['solved'] for r in rows)/len(rows), 'rows': rows}


def train(tasks, out, loops, seed, steps, threads=2):
    torch.set_num_threads(threads); torch.manual_seed(seed)
    random.seed(seed); torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    batch, y = examples([t for t in tasks if t['split'] == 'train'])
    model = ProofPolicy(loops); optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    g = torch.Generator().manual_seed(seed + 1000); losses = []
    for step in range(steps):
        ix = torch.randint(len(y), (64,), generator=g)
        b = tuple(x[ix] for x in batch); target = y[ix]
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = objective(model(b), b, target); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 50 == 0 or step == steps-1: losses.append([step, float(loss.detach())])
    model.eval()
    val = evaluate(model, [t for t in tasks if t['split'] == 'val'])
    tag = f'l{loops}_s{seed}_u{steps}'
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f'{tag}.pt'; torch.save(model.state_dict(), ckpt)
    summary = {'tag': tag, 'loops': loops, 'seed': seed, 'steps': steps,
               'parameters': sum(p.numel() for p in model.parameters()), 'train_states': len(y),
               'losses': losses, 'validation': {k:v for k,v in val.items() if k != 'rows'},
               'elapsed_seconds': time.perf_counter()-started,
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'torch_version': torch.__version__, 'threads': threads,
               'checkpoint_sha256': hashlib.sha256(ckpt.read_bytes()).hexdigest()}
    (out / f'{tag}.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def final_eval(tasks, out, artifacts):
    torch.set_num_threads(2); results = []
    tests = {s:[t for t in tasks if t['split']==s] for s in ['test','depth_ood']}
    for file in sorted(out.glob('l*_s*_u*.json')):
        meta = json.loads(file.read_text()); model = ProofPolicy(meta['loops'])
        model.load_state_dict(torch.load(out / (meta['tag'] + '.pt'), weights_only=True)); model.eval()
        settings = [('trained_depth', meta['loops'], False)]
        if meta['loops'] == 4:
            settings += [('goal_removed',4,True), ('extra_loops_8',8,False)]
        for label, loops, no_goal in settings:
            entry = {'tag':meta['tag'], 'condition':label, 'loops':loops, 'seed':meta['seed'],
                     'parameters':meta['parameters'], 'steps':meta['steps']}
            for split, ts in tests.items():
                entry[split] = evaluate(model, ts, loops, no_goal, seed=meta['seed'])
                if label == 'trained_depth':
                    valid = [(t, row) for t, row in zip(ts, entry[split]['rows']) if row['solved']]
                    text = 'set_option linter.unusedVariables false\n\n' + '\n\n'.join(
                        lean_theorem(t, 'pred_' + t['id'], row['trace']) for t, row in valid) + '\n'
                    (artifacts / f'Predictions_{meta["tag"]}_{split}.lean').write_text(text)
            results.append(entry)
    for baseline in ['random','direct_goal','oracle']:
        for seed in [17,29,43]:
            entry = {'tag':baseline, 'condition':'baseline','seed':seed,'parameters':0}
            for split, ts in tests.items(): entry[split] = evaluate(None, ts, baseline=baseline, seed=seed)
            results.append(entry)
    (out / 'evaluation.json').write_text(json.dumps(results, indent=2)+'\n')
    for r in results:
        print(r['tag'], r['condition'], 'ID',r['test']['solved'], '/',r['test']['total'],
              'OOD',r['depth_ood']['solved'],'/',r['depth_ood']['total'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['train','evaluate'])
    p.add_argument('--data', type=Path, default=Path(__file__).parent/'artifacts')
    p.add_argument('--out', type=Path, default=Path(__file__).parent/'results')
    p.add_argument('--loops', type=int, default=4); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--steps',type=int,default=300); p.add_argument('--threads',type=int,default=2)
    a=p.parse_args(); tasks=[json.loads(x) for x in (a.data/'tasks.jsonl').read_text().splitlines()]
    if a.command == 'train': train(tasks,a.out,a.loops,a.seed,a.steps,a.threads)
    else: final_eval(tasks,a.out,a.data)
