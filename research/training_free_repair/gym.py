"""TFL-02/06: frozen graph decision repair; rank-one BPD precursor.
Development only. Run --self-test for structural checks, or --model Qwen/Qwen3-0.6B.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

ARMS = ('native', 'repeat', 'anchor', 'ablate', 'sham', 'component', 'sham_component')
LABELS = 'ABCD'


def root(seed):
    rng = random.Random(seed)
    names = [f'n{i:02d}' for i in rng.sample(range(100), 42)]
    start, goal = names[:2]
    branches = [names[2+10*i:12+10*i] for i in range(4)]
    gold = rng.randrange(4)
    edges = []
    for i, b in enumerate(branches):
        chain = b[:rng.randint(3, 10)]
        edges += [(start, chain[0])] + list(zip(chain, chain[1:]))
        if i == gold:
            edges.append((chain[-1], goal))
        edges += [(chain[j], chain[k]) for j in range(len(chain))
                  for k in range(j+2, len(chain)) if rng.random() < .12]
    rng.shuffle(edges)
    choices = [b[0] for b in branches]
    def reaches(node):
        seen, todo = set(), [node]
        while todo:
            x = todo.pop()
            if x == goal:
                return True
            if x not in seen:
                seen.add(x)
                todo.extend(v for u, v in edges if u == x)
        return False
    valid = [i for i, n in enumerate(choices) if reaches(n)]
    assert valid == [gold]
    prompt = (f'Directed graph. Start {start}; target {goal}. Edges: ' +
              ', '.join(f'{u}->{v}' for u, v in edges) +
              '. Which first move permits reaching the target? ' +
              ' '.join(f'{a}: {n}' for a, n in zip(LABELS, choices)) +
              '. Answer with one letter. Answer:')
    return {'root_id': str(seed), 'prompt': prompt, 'gold': gold}


def anchor(x, y, g):
    mid = x + (y-x)*.5
    end = mid + (g(mid)-mid)*.5
    return (y+end)*.5


def intervene(y, a, arm, v, seed):
    # Only the final prompt position is edited; the anchor's other positions stay intact.
    if arm == 'anchor':
        return a, 0.
    d = (a-y)[0, -1].float()
    v = v.to(d)
    p = torch.dot(d, v)*v
    q = torch.randn(v.shape, generator=torch.Generator().manual_seed(seed)).to(v)
    q = q-torch.dot(q, v)*v
    q = q/q.norm().clamp_min(1e-12)*p.norm()
    base, edit = {'ablate': (a, -p), 'sham': (a, -q),
                  'component': (y, p), 'sham_component': (y, q)}[arm]
    out = base.clone()
    out[0, -1] = (base[0, -1].float()+edit).to(base.dtype)
    return out, float((out[0, -1].float()-base[0, -1].float()).norm())


class Adapter:
    def __init__(self, layer):
        self.layer = layer
        self.arm, self.v, self.seed = 'native', None, 0
        self.handle = layer.register_forward_hook(self.hook, with_kwargs=True)

    def hook(self, mod, args, kwargs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError('Expected tensor-returning decoder layer; pin Transformers 4.57.1')
        self.calls = 1
        self.delta, self.dose = None, 0.
        if self.arm == 'native':
            return output
        x = args[0] if args else kwargs['hidden_states']
        def g(h):
            self.calls += 1
            return mod.forward(h, *args[1:], **kwargs) if args else mod.forward(**dict(kwargs, hidden_states=h))
        if self.arm == 'repeat':
            z = g(x)
            if not torch.equal(z, output):
                raise RuntimeError('Repeat control differs from native output')
            return output
        a = anchor(x, output, g)
        self.delta = (a-output)[0, -1].detach().float().cpu()
        result, self.dose = intervene(output, a, self.arm, self.v, self.seed)
        return result


def paired(rows, candidate, baseline):
    b = sum(r['correct'][candidate] and not r['correct'][baseline] for r in rows)
    c = sum(r['correct'][baseline] and not r['correct'][candidate] for r in rows)
    return {'repair': b, 'damage': c, 'net': (b-c)/len(rows)}


def self_test():
    tasks = [root(i) for i in range(256)]
    assert len({r['prompt'] for r in tasks}) == 256
    assert all(root(i) == tasks[i] for i in range(256))
    class Block(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + .2*hidden_states.square() + .1
    layer = Block()
    h = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(1))
    native = layer(h)
    expected = anchor(h, native, layer.forward)
    adapter = Adapter(layer)
    assert torch.equal(layer(h), native) and adapter.calls == 1
    adapter.arm = 'repeat'
    assert torch.equal(layer(h), native) and adapter.calls == 2
    adapter.arm = 'anchor'
    assert torch.equal(layer(h), expected) and adapter.calls == 2
    adapter.v = adapter.delta / adapter.delta.norm()
    edits = {}
    for arm in ARMS[3:]:
        adapter.arm = arm
        out = layer(h)
        assert adapter.calls == 2
        edits[arm] = adapter.dose
        ref = expected if arm in ('ablate', 'sham') else native
        assert torch.equal(out[:, :-1], ref[:, :-1])
    assert max(edits.values())-min(edits.values()) < 1e-6
    adapter.handle.remove()
    assert torch.equal(layer(h), native)
    print(json.dumps({'scope': 'structural CPU checks only', 'roots': len(tasks), 'matched_doses': edits}))


@torch.inference_mode()
def run(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    dtype = torch.float32 if args.device == 'cpu' else torch.float16
    load = {'revision': args.revision, 'torch_dtype': dtype}
    if args.nf4:
        if args.device != 'cuda':
            raise ValueError('NF4 requires CUDA')
        load.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype), device_map={'': 0})
    model = AutoModelForCausalLM.from_pretrained(args.model, **load).eval().requires_grad_(False)
    if not args.nf4:
        model.to(args.device)
    adapter = Adapter(model.model.layers[args.layer])
    def forward(r, arm):
        adapter.arm, adapter.seed = arm, int(r['root_id'])
        ids = tok.encode(r['prompt'], add_special_tokens=False)
        labels = [tok.encode(r['prompt']+' '+s, add_special_tokens=False) for s in LABELS]
        if not all(x[:-1] == ids and len(x) == len(ids)+1 for x in labels) or len({x[-1] for x in labels}) != 4:
            raise ValueError('Answer labels must be distinct single-token suffixes')
        logits = model(input_ids=torch.tensor([ids], device=args.device), use_cache=False).logits[0, -1].float()
        scores = logits[[x[-1] for x in labels]].cpu()
        return scores.tolist(), adapter.delta, adapter.dose, adapter.calls
    try:
        cal = [root(i) for i in range(args.cal)]
        deltas = torch.stack([forward(r, 'anchor')[1] for r in cal])
        if not torch.isfinite(deltas).all() or deltas.norm() == 0:
            raise RuntimeError('Degenerate calibration deltas')
        adapter.v = torch.linalg.svd(deltas, full_matrices=False).Vh[0]
        rows = []
        for i in range(100000, 100000+args.dev):
            r = root(i)
            scores, dose, calls, correct = {}, {}, {}, {}
            for arm in ARMS:
                s, _, dose[arm], calls[arm] = forward(r, arm)
                scores[arm] = s
                correct[arm] = int(max(range(4), key=lambda j: s[j]) == r['gold'])
            if scores['native'] != scores['repeat']:
                raise RuntimeError('Native/repeat logits diverged')
            rows.append(dict(r, scores=scores, correct=correct, dose=dose, calls=calls))
    finally:
        adapter.handle.remove()
    payload = {'experiment': 'TFL-02/06', 'verdict': 'REDESIGN_OR_PILOT_ONLY',
               'model': args.model, 'revision': model.config._commit_hash, 'layer': args.layer,
               'precision': 'nf4' if args.nf4 else str(dtype), 'unit': 'independent seeded graph root within one generator',
               'calibration_roots': args.cal, 'development_roots': args.dev, 'sealed_readout': False,
               'minimum_effect': .02, 'planned_comparisons': 3,
               'contrasts': {f'{a}_vs_{b}': paired(rows, a, b) for a,b in [('anchor','repeat'),('sham','ablate'),('component','sham_component')]},
               'direction': adapter.v.tolist(), 'rows': rows}
    raw = json.dumps(payload, sort_keys=True, indent=2)+'\n'
    args.out.write_text(raw)
    print(hashlib.sha256(raw.encode()).hexdigest(), args.out, payload['contrasts'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p.add_argument('--revision', default='main')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--nf4', action='store_true')
    p.add_argument('--layer', type=int, default=13)
    p.add_argument('--cal', type=int, default=64)
    p.add_argument('--dev', type=int, default=256)
    p.add_argument('--out', type=Path, default=Path('development.json'))
    a = p.parse_args()
    if a.cal < 1 or a.cal >= 100000 or a.dev < 1:
        p.error('invalid calibration/development counts')
    self_test() if a.self_test else run(a)
