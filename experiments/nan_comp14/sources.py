"""Outcome-blind fresh paired worlds; only named exposed fixtures excluded."""
from collections import Counter
import hashlib
import json
import random
from .common import ROOT, read, rows, sha, write_new

NODES = 'ABCDEFGHIJKLMNOP'


def demo(rng, hops):
    values = list(NODES); rng.shuffle(values)
    edges = dict(zip(NODES, values)); start = rng.choice(NODES); answer = start
    for _ in range(hops):
        answer = edges[answer]
    order = list(NODES); rng.shuffle(order)
    return question(edges, order, start, hops) + ' ' + answer


def question(edges, order, start, hops=1):
    line = '[not provided]' if edges is None else ', '.join(k + '->' + edges[k] for k in order)
    return f'Edges: {line}\nStart: {start}\nHops: {hops}\nAnswer:'


def materialize(cfg):
    public = []; host = []
    for split, count in cfg['roots'].items():
        for index in range(count):
            seed = int.from_bytes(hashlib.sha256(f"{cfg['source_seed']}:{split}:{index}".encode()).digest()[:8], 'big')
            rng = random.Random(seed); query = rng.choice(NODES)
            a, b = NODES[2 * (index % 8):2 * (index % 8) + 2]
            destinations = list(NODES); rng.shuffle(destinations)
            edges = dict(zip(NODES, destinations))
            owner_a = next(k for k in NODES if edges[k] == a)
            edges[query], edges[owner_a] = edges[owner_a], edges[query]
            other = next(k for k in NODES if edges[k] == b)
            counterfactual = dict(edges)
            counterfactual[query], counterfactual[other] = edges[other], edges[query]
            assert edges[query] == a and counterfactual[query] == b
            order = list(NODES); rng.shuffle(order)
            demonstration_seed = rng.getrandbits(64)
            legacy_rng = random.Random(demonstration_seed)
            focused_rng = random.Random(demonstration_seed)
            demonstrations = {
                'legacy': [demo(legacy_rng, h) for h in (1, 2, 3)],
                'focused': [demo(focused_rng, 1) for _ in range(3)]}
            instructions = {
                'legacy': 'Follow the directed edges exactly the requested number of hops. Reply only the final node A through P.',
                'focused': 'Read exactly ONE edge whose left side matches Start. Copy its right-hand node. Reply with one letter A through P and nothing else.'}
            root = f'comp14:{split}:{index:03d}'
            for world, graph, answer in [('A', edges, a), ('B', counterfactual, b)]:
                for arm in cfg['arms']:
                    template = arm.replace('_erased', '')
                    if arm == 'copy':
                        prompt = 'Copy the following verified answer exactly, with no explanation:\n' + answer
                    else:
                        body = question(None if arm.endswith('_erased') else graph, order, query)
                        prompt = instructions[template] + '\n\n' + '\n\n'.join(demonstrations[template] + [body])
                    key = root + ':' + world + ':' + arm
                    public.append(dict(id=key, root=root, split=split, world=world, arm=arm, prompt=prompt))
                    host.append(dict(id=key, root=root, split=split, world=world, arm=arm, answer=answer,
                                     query=query, edges=graph, paired_changed_sources=[query, other]))
    return public, host


def main():
    cfg = read(ROOT / 'config.json'); public, host = materialize(cfg)
    folder = ROOT / 'fixtures'
    if folder.exists():
        raise FileExistsError('Preserve the original fresh source draw')
    old_paths = ['experiments/nan_gym05/fixtures/public.jsonl',
                 'experiments/nan_fit07/fixtures/public.jsonl',
                 'experiments/nan_ctx08/fixtures/public.jsonl']
    old = {r['prompt'] for p in old_paths for r in rows(p)}
    main_prompts = [r['prompt'] for r in public if r['arm'] in cfg['templates']]
    assert len(main_prompts) == len(set(main_prompts)) and not old.intersection(main_prompts)
    for split in cfg['roots']:
        answers = Counter(r['answer'] for r in host if r['split'] == split and r['arm'] == 'focused')
        assert set(answers) == set(NODES) and len(set(answers.values())) == 1
    folder.mkdir()
    for name, values in [('public.jsonl', public), ('host_labels.jsonl', host)]:
        with (folder / name).open('x', encoding='utf-8', newline='\n') as f:
            f.write(''.join(json.dumps(v, separators=(',', ':')) + '\n' for v in values))
    write_new(folder / 'source_receipt.json', dict(contexts=len(public), roots=cfg['roots'],
        source_only=True, model_outcomes_used=False, main_prompt_overlap_with_named_exposed_sources=0,
        exposed_exclusion_files_sha256={p: sha(p) for p in old_paths},
        source_rule='Two permutation worlds differ by swapping exactly two outgoing destinations; query and row order fixed; both answers differ.',
        answer_balance='Every label occurs equally within each split, in each template and world-pair aggregate.',
        repeated_copy_and_erased_prompts_are_controls=True, sealed_test_accessed_or_used_for_exclusions=False))
    print({'contexts': len(public), 'roots': cfg['roots'], 'main_prompts': len(main_prompts)})


if __name__ == '__main__':
    main()
