"""Generate fresh worlds and select record counts using tokenizer lengths only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from .common import ROOT, read, sha, write

NODES = 'ABCDEFGHIJKLMNOP'
INSTRUCTION = ('Read exactly ONE edge in the graph named by Graph in the question. '
    'Match the left side to Start and copy its right-hand node. Other graphs are independent. '
    'Reply with one letter A through P and nothing else. A question mark means that edge is unavailable.\n\n'
    'Example: Graph EX1: A->D, D->M, M->A.\nGraph: EX1\nStart: A\nAnswer: D\n\n'
    'Example: Graph EX2: B->P, P->C, C->B.\nGraph: EX2\nStart: P\nAnswer: C\n\n'
    'Example: Graph EX3: F->K, K->H, H->F.\nGraph: EX3\nStart: H\nAnswer: F\n\nGraphs:\n')


def draw(cfg, split, index):
    seed = int.from_bytes(hashlib.sha256(f"{cfg['source_seed']}:{split}:{index}".encode()).digest()[:8], 'big')
    rng = random.Random(seed); start = rng.choice(NODES)
    a, b = NODES[2 * (index % 8):2 * (index % 8) + 2]
    values = list(NODES); rng.shuffle(values); graph = dict(zip(NODES, values))
    owner = next(k for k in NODES if graph[k] == a)
    graph[start], graph[owner] = graph[owner], graph[start]
    other = next(k for k in NODES if graph[k] == b)
    changed = dict(graph); changed[start], changed[other] = graph[other], graph[start]
    order = list(NODES); rng.shuffle(order)
    identifiers = ['G' + str(v) for v in rng.sample(range(100000, 1000000), cfg['maximum_graph_records'] + 1)]
    distractors = []
    for name in identifiers[1:]:
        v = list(NODES); rng.shuffle(v); keys = list(NODES); rng.shuffle(keys)
        distractors.append((name, dict(zip(NODES, v)), keys))
    return dict(root=f'ctx21:{split}:{index:03d}', split=split, target_id=identifiers[0], start=start,
        worlds={'A': graph, 'B': changed}, answers={'A': a, 'B': b}, order=order,
        distractors=distractors, changed_sources=[start, other])


def build(source, count, position, world, arm):
    assert count >= 3 and arm in ('main', 'copy', 'erased')
    target_index = {'early': 0, 'middle': count // 2, 'late': count - 1}[position]
    records = list(source['distractors'][:count - 1])
    graph = {k: '?' for k in NODES} if arm == 'erased' else source['worlds'][world]
    records.insert(target_index, (source['target_id'], graph, source['order']))
    text = INSTRUCTION + '\n'.join('Graph ' + name + ': ' + ', '.join(k+'->'+g[k] for k in order) + '.' for name, g, order in records)
    text += '\n\nGraph: ' + source['target_id'] + '\nStart: ' + source['start'] + '\n'
    if arm == 'copy':
        text += 'Copy control: ignore the question and repeat exactly this supplied label: ' + source['answers'][world] + '\n'
    return text + 'Answer:', target_index


def main():
    p = argparse.ArgumentParser(); p.add_argument('--snapshot', required=True); a = p.parse_args()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ.pop('TRANSFORMERS_CACHE', None)
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    from transformers import PreTrainedTokenizerFast
    from experiments.nan_gym05.common import prompt_ids
    snapshot = Path(a.snapshot)
    tok = SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(snapshot / 'tokenizer.json')))
    cfg = read(ROOT / 'config.json'); folder = ROOT / 'fixtures'
    assert not folder.exists(), 'Preserve the original source draw; no outcome-based source regeneration'
    sources = {(split, i): draw(cfg, split, i) for split, n in cfg['roots'].items() for i in range(n)}
    public = []; host = []; token_rows = {}; selections = []
    for budget in cfg['context_budgets']:
        for (split, index), source in sources.items():
            positions = ['middle'] if split == 'calibration' else cfg['positions']
            def lengths(count):
                return [len(prompt_ids(tok, snapshot, build(source, count, pos, w, arm)[0]))
                    for pos in cfg['positions'] for w in ('A', 'B') for arm in cfg['arms']]
            # Fixed token-only binary search; no claim of a global optimum across
            # token-boundary merges. Every final variant must fit the length band.
            best = None; lo = 3; hi = cfg['maximum_graph_records']; checked = []
            while lo <= hi:
                count = (lo + hi) // 2
                values = lengths(count)
                checked.append(dict(graph_records=count, minimum_tokens=min(values), maximum_tokens=max(values)))
                if max(values) <= budget:
                    best = (count, min(values), max(values))
                    lo = count + 1
                else:
                    hi = count - 1
            assert best is not None
            count, smallest, largest = best
            assert budget - cfg['maximum_underfill_tokens'] <= smallest <= largest <= budget, (source['root'], budget, best)
            selections.append(dict(root=source['root'], context_budget=budget, graph_records=count,
                minimum_all_variant_tokens=smallest, maximum_all_variant_tokens=largest, search_trace=checked))
            for pos in positions:
                for world in ('A', 'B'):
                    for arm in cfg['arms']:
                        text, target_index = build(source, count, pos, world, arm)
                        ids = prompt_ids(tok, snapshot, text)
                        key = f"{source['root']}:{budget}:{pos}:{world}:{arm}"
                        public.append(dict(id=key, root=source['root'], split=split, context_budget=budget, position=pos,
                            world=world, arm=arm, graph_records=count, target_graph_index=target_index, prompt=text))
                        host.append(dict(id=key, answer=source['answers'][world], query=source['start'],
                            target_graph_id=source['target_id'], target_edges=source['worlds'][world], changed_sources=source['changed_sources']))
                        token_rows[key] = dict(tokens=len(ids), ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            print(dict(source=source['root'], budget=budget, count=count, token_range=[smallest, largest]), flush=True)
    assert len(public) == 624 and len(token_rows) == len(public)
    folder.mkdir()
    for name, values in [('public.jsonl', public), ('host_labels.jsonl', host)]:
        with (folder / name).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(''.join(json.dumps(v, separators=(',', ':'))+'\n' for v in values))
    write(folder / 'tokenizer_receipt.json', dict(contexts=token_rows, selections=selections,
        tokenizer_files_sha256={name:sha(snapshot/name) for name in ('tokenizer.json', 'tokenizer_config.json')},
        public_sha256=sha(folder/'public.jsonl'), source_only=True, pretrained_weights_loaded=False,
        rule='Fixed token-only binary search3..160 over longest registered variant; final variants all within the fixed band; no global-maximum claim or model outcomes.',
        sealed_test_accessed_or_used_as_exclusion=False))


if __name__ == '__main__':
    main()
