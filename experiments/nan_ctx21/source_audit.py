"""Independent textual source/oracle/erasure/roster checks; no model imports."""
from collections import Counter, defaultdict
import re
from .common import ROOT, read, rows, sha, write

NODES = set('ABCDEFGHIJKLMNOP')
EXPOSED = ('experiments/nan_comp14/fixtures/public.jsonl', 'experiments/nan_gym05/fixtures/public.jsonl')


def parse(row):
    question = re.search(r'\n\nGraph: (G[0-9]{6})\nStart: ([A-P])\n', row['prompt'])
    assert question and row['prompt'].endswith('\nAnswer:')
    graph_id, start = question.groups()
    lines = re.findall(r'^Graph (G[0-9]{6}): (.+)\.$', row['prompt'], flags=re.MULTILINE)
    assert len(lines) == row['graph_records'] and len({name for name, _ in lines}) == len(lines)
    graphs = {}
    for name, text in lines:
        edges = re.findall(r'([A-P])->([A-P?])', text)
        assert len(edges) == 16 and {k for k, _ in edges} == NODES
        graph = dict(edges); graphs[name] = graph
        if name == graph_id and row['arm'] == 'erased':
            assert set(graph.values()) == {'?'}
        else:
            assert set(graph.values()) == NODES
    index = [name for name, _ in lines].index(graph_id)
    assert index == row['target_graph_index'] == {'early':0, 'middle':len(lines)//2, 'late':len(lines)-1}[row['position']]
    return graph_id, start, graphs, lines


def old_graphs():
    values = set()
    for file in EXPOSED:
        for row in rows(file):
            for text in re.findall(r'Edges: ([^\n]+)', row['prompt']):
                edges = re.findall(r'([A-P])->([A-P])', text)
                if len(edges) == 16 and len(dict(edges)) == 16:
                    values.add(tuple(sorted(edges)))
    return values


def validate(public, host, tokens, cfg, exposed=()):
    gold = {r['id']:r for r in host}; assert len(gold) == len(host) == len(public) == 624
    assert len({r['id'] for r in public}) == len(public) and set(gold) == {r['id'] for r in public} == set(tokens['contexts'])
    pairs = defaultdict(dict); root_graphs = defaultdict(set); graphs_seen = set(); counts = Counter()
    expected = []
    for budget in cfg['context_budgets']:
        for split, n in cfg['roots'].items():
            for i in range(n):
                root = f'ctx21:{split}:{i:03d}'
                for pos in (['middle'] if split == 'calibration' else cfg['positions']):
                    for world in ('A', 'B'):
                        for arm in cfg['arms']:
                            expected.append(f'{root}:{budget}:{pos}:{world}:{arm}')
    assert [r['id'] for r in public] == expected
    for row in public:
        name, start, graphs, lines = parse(row); label = gold[row['id']]
        assert label['query'] == start and label['target_graph_id'] == name and label['answer'] in NODES
        assert set(label['target_edges']) == set(label['target_edges'].values()) == NODES
        assert label['target_edges'][start] == label['answer']
        if row['arm'] == 'main':
            assert graphs[name] == label['target_edges']
        elif row['arm'] == 'copy':
            assert graphs[name] == label['target_edges']
            assert row['prompt'].endswith('Copy control: ignore the question and repeat exactly this supplied label: '+label['answer']+'\nAnswer:')
        else:
            assert row['arm'] == 'erased' and set(graphs[name].values()) == {'?'}
        root_graphs[row['root'], row['world']].add(tuple(sorted(label['target_edges'].items())))
        graphs_seen.add(tuple(sorted(label['target_edges'].items())))
        t = tokens['contexts'][row['id']]
        assert row['context_budget'] - cfg['maximum_underfill_tokens'] <= t['tokens'] <= row['context_budget']
        assert re.fullmatch('[0-9a-f]{64}', t['ids_sha256'])
        pairs[row['root'], row['context_budget'], row['position'], row['arm']][row['world']] = (row, label, graphs, lines)
        counts[row['split'], row['context_budget'], row['position'], row['arm']] += 1
    assert len(root_graphs) == len(graphs_seen) == 40 and all(len(v) == 1 for v in root_graphs.values())
    assert not graphs_seen.intersection(exposed)
    for by_world in pairs.values():
        assert set(by_world) == {'A', 'B'}
        a, b = by_world['A'], by_world['B']; assert a[1]['answer'] != b[1]['answer']
        assert a[1]['changed_sources'] == b[1]['changed_sources']
        changed = [k for k in sorted(NODES) if a[1]['target_edges'][k] != b[1]['target_edges'][k]]
        assert set(changed) == set(a[1]['changed_sources']) and len(changed) == 2 and a[1]['query'] in changed
        target = a[1]['target_graph_id']; assert [v[0] for v in a[3]] == [v[0] for v in b[3]]
        assert {k:v for k,v in a[2].items() if k != target} == {k:v for k,v in b[2].items() if k != target}
        if a[0]['arm'] == 'erased':
            assert a[0]['prompt'] == b[0]['prompt']
    for (split, budget, pos, arm), count in counts.items():
        assert count == 2 * cfg['roots'][split]
    balance = Counter(gold[r['id']]['answer'] for r in public if r['split'] == 'development' and r['context_budget'] == 4096 and r['position'] == 'early' and r['arm'] == 'main')
    assert set(balance) == NODES and set(balance.values()) == {2}
    assert len(tokens['selections']) == 40
    for entry in tokens['selections']:
        trace = entry['search_trace']; assert len({v['graph_records'] for v in trace}) == len(trace) <= 8
        lo, hi, selected = 3, cfg['maximum_graph_records'], None
        for item in trace:
            count = (lo + hi) // 2; assert item['graph_records'] == count and lo <= hi
            assert item['minimum_tokens'] <= item['maximum_tokens']
            if item['maximum_tokens'] <= entry['context_budget']:
                selected = item; lo = count + 1
            else:
                hi = count - 1
        assert lo > hi and selected and selected['graph_records'] == entry['graph_records']
        assert selected['minimum_tokens'] == entry['minimum_all_variant_tokens'] and selected['maximum_tokens'] == entry['maximum_all_variant_tokens']
        matching = [r for r in public if r['root'] == entry['root'] and r['context_budget'] == entry['context_budget']]
        assert all(r['graph_records'] == entry['graph_records'] for r in matching)
    return dict(status='passed', contexts=624, CAL_contexts=48, DEV_contexts=576, fresh_target_graphs=40,
        independent_roots=cfg['roots'], named_exposed_graphs=len(exposed), source_overlap=0,
        erased_world_prompts_byte_identical=True, all_target_destinations_erased=True,
        DEV_pair_answer_counts=dict(sorted(balance.items())), tokenizer_search_traces_checked=40,
        source_only=True, pretrained_model_loaded=False, sealed_test_accessed_or_used_as_exclusion=False)


def main():
    folder = ROOT / 'fixtures'
    result = validate(rows(folder/'public.jsonl'), rows(folder/'host_labels.jsonl'), read(folder/'tokenizer_receipt.json'), read(ROOT/'config.json'), old_graphs())
    result['named_exposed_files_sha256'] = {name:sha(name) for name in EXPOSED}
    result['scope'] = 'Independent textual oracle, full-target erasure, CF pair differences, fixed whole-roster ordering and token-only search-trace arithmetic; no model outcomes.'
    write(folder/'independent_source_audit.json', result)
    print(result)


if __name__ == '__main__':
    main()
