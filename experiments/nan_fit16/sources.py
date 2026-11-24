"""Reuse all focused CAL/DEV worlds plus all copy/erasure controls, without outcomes."""
from .common import ROOT, read, rows, sha, write


def main():
    source = rows('experiments/nan_comp14/fixtures/public.jsonl')
    labels = {r['id']: r for r in rows('experiments/nan_comp14/fixtures/host_labels.jsonl')}
    public = [dict(r, upstream_ordinal=i, copy_control=r['arm'] != 'focused', schema='node')
              for i, r in enumerate(source) if r['arm'] in ('focused', 'copy', 'focused_erased')]
    assert len(public) == 144 and len({r['root'] for r in public}) == 24
    cal = [r for r in public if r['split'] == 'calibration' and r['arm'] == 'focused']
    assert len(cal) == 16 and len({r['root'] for r in cal}) == 8
    for path, records in [('public.jsonl', public), ('host_labels.jsonl', [labels[r['id']] for r in public])]:
        target = ROOT / 'fixtures' / path; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as f:
            import json
            f.write(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in records))
    write(ROOT / 'fixtures/source_receipt.json', dict(contexts=144, CAL_roots=8, DEV_roots=16,
        CAL_training_worlds=16, order='Original source order; both worlds of every CAL root; no individual correctness filtering.',
        upstream_public_sha256=sha('experiments/nan_comp14/fixtures/public.jsonl'),
        upstream_host_oracle_sha256=sha('experiments/nan_comp14/fixtures/host_labels.jsonl'),
        source_public_sha256=sha(ROOT / 'fixtures/public.jsonl'),
        source_host_oracle_sha256=sha(ROOT / 'fixtures/host_labels.jsonl'),
        exposed=True, sealed_test_accessed=False))


if __name__ == '__main__':
    main()
