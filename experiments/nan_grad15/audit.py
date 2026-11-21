"""Archive audit: exact all-one closure, finite gradients and independent KL."""
import argparse
import math
from pathlib import Path
from .common import ROOT, RUNS, read, sha, verify, write


def kl(prediction, target):
    import numpy as np
    def lp(value):
        value = value.astype(np.float64); shifted = value - value.max()
        return shifted - np.log(np.exp(shifted).sum())
    p, q = lp(target), lp(prediction)
    return float((np.exp(p) * (p - q)).sum())


def run(backend):
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); verify()
    import numpy as np
    base = RUNS / backend; cfg = read(ROOT / 'config.json'); terminal = read(base / 'receipt.json')
    source = read(ROOT / 'fixtures/public.json'); manifest = read(base / 'manifest.json') if (base / 'manifest.json').exists() else None
    baseline = {}
    for i, row in enumerate(source):
        p = base / f'baseline_{i:02d}.json'
        if not p.exists():
            continue
        r = read(p); assert r['source_id'] == row['id'] and r['upstream_ordinal'] == row['upstream_ordinal']
        archive = base / r['archive']['file']; assert sha(archive) == r['archive']['sha256']
        upstream = Path('runs/nan_comp14') / backend / f"case_{row['upstream_ordinal']:04d}.json"
        assert sha(upstream) == r['upstream_case_sha256']; old = read(upstream)
        with np.load(archive, allow_pickle=False) as z:
            baseline[i] = {k: z[k].copy() for k in z.files}
            for t in (0, 1):
                logits = z[f'logits_{t}']; assert logits.shape == (166144,) and np.isfinite(logits).all()
                score = old['initial_phases'][t]['score']
                assert int(logits.argmax()) == score['free_top_token_id'] and float(logits.max()) == score['free_top_token_logit']
                x = logits.astype(np.float64); m = float(x.max()); normalizer = m + math.log(float(np.exp(x - m).sum()))
                assert abs(normalizer - score['full_log_normalizer']) <= 1e-9
                assert z[f'boundary_{t}'].shape == (3072,) and np.isfinite(z[f'boundary_{t}']).all()
    results = []
    for condition in terminal['conditions']:
        passes = condition['passes']; passed = 0; closed = 0
        for i in range(condition['cases']):
            r = read(base / f't{passes}_case_{i:02d}.json'); assert r['source_id'] == source[i]['id']
            archive = base / r['archive']['file']; assert sha(archive) == r['archive']['sha256']; closed += 1
            stats = r['saved_tensor_statistics']; assert stats['maximum_added_cpu_bytes'] == cfg['maximum_added_saved_tensor_CPU_bytes']
            assert 0 <= stats['peak_added_cpu_bytes'] <= cfg['maximum_added_saved_tensor_CPU_bytes']
            with np.load(archive, allow_pickle=False) as z:
                if r['status'] == 'passed':
                    assert r['all_one_closure_bit_exact']
                    for name, shape in [('V', (10752, 8)), ('U', (8, 3072))]:
                        gradient = z[name + '_gradient']; assert gradient.shape == shape and np.isfinite(gradient).all()
                        norm = float(np.linalg.norm(gradient.astype(np.float64)))
                        assert norm > 0 and math.isclose(norm, r['gradient_L2'][name], rel_tol=2e-5, abs_tol=1e-8)
                    losses = []
                    for t in range(passes):
                        assert np.array_equal(z[f'clean_logits_{t}'], baseline[i][f'logits_{t}'])
                        assert np.array_equal(z[f'clean_logits_{t}'], z[f'all_one_logits_{t}'])
                        assert np.array_equal(z[f'clean_boundary_{t}'], z[f'all_one_boundary_{t}'])
                        assert z[f'masked_logits_{t}'].shape == (166144,) and z[f'masked_boundary_{t}'].shape == (3072,)
                        assert np.isfinite(z[f'masked_logits_{t}']).all() and np.isfinite(z[f'masked_boundary_{t}']).all()
                        losses.append(kl(z[f'masked_logits_{t}'], z[f'clean_logits_{t}']))
                    assert math.isclose(float(np.mean(losses)), r['loss'], rel_tol=2e-4, abs_tol=5e-5)
                    passed += 1
            assert r['status'] in ('passed', 'failed')
        if condition['status'] == 'passed':
            assert closed == passed == 4
        results.append(dict(passes=passes, condition_status=condition['status'], archived_cases=closed,
                            successful_gradient_cases=passed, missing_registered_cases=4 - closed))
    if terminal['status'] == 'completed':
        assert manifest and manifest['target_parameter_sha256'] == terminal['target_parameter_sha256_unchanged']
        assert len(baseline) == 4 and len(results) == 2
    output = dict(status='passed', backend=backend, worker_status=terminal['status'], conditions=results,
        source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), receipt_sha256=sha(base / 'receipt.json'),
        source_roots=2, CAL_prefixes_verified=len(baseline), optimizer_steps=0, factors_trained=False,
        limitation='Archives, closure and gradient finite/nonzero/norm contracts checked; no second full model backward or finite-difference gradient replay.',
        sealed_test_accessed=False)
    write(RUNS / 'audits' / (backend + '.json'), output)
    print(output, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True); a = p.parse_args(); run(a.backend)
