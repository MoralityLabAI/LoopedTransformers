"""Fixed source-order CAL prefixes; never select successful model cases."""
from .common import ROOT, read, rows, sha, write


def select(public):
    roots = sorted({v['root'] for v in public if v['split'] == 'calibration'})[:2]
    chosen = [dict(v, upstream_ordinal=i) for i, v in enumerate(public)
              if v['root'] in roots and v['split'] == 'calibration' and v['arm'] == 'focused']
    assert len(chosen) == 4 and {v['world'] for v in chosen} == {'A', 'B'}
    return chosen


def main():
    path = 'experiments/nan_comp14/fixtures/public.jsonl'
    chosen = select(rows(path)); folder = ROOT / 'fixtures'; folder.mkdir(exist_ok=False)
    write(folder / 'public.json', chosen)
    write(folder / 'source_receipt.json', dict(upstream_public_sha256=sha(path), contexts=4, CAL_roots=2,
        selection='First two CAL source roots, both worlds, focused recipe; independent of outcomes.',
        source_exposure='COMP14 NF4 aggregates exposed; prefixes reused for engineering, not fresh confirmation.',
        DEV_or_copy_contexts_selected=0, model_correctness_used_for_selection=False, sealed_test_accessed=False))
    print({'fixed_CAL_prefixes': 4, 'source_roots': 2})


if __name__ == '__main__':
    main()
