"""Independent arithmetic audit of archived codes, routes and physical geometry."""
import argparse
from collections import defaultdict
from pathlib import Path
from .common import ROOT, RUNS, read, sha, verify, write
from .cache import load, STRATA


def macro(values, indices, meta):
    import numpy as np
    roots = sorted({meta[i]['root'] for i in indices})
    return np.asarray([np.asarray([values[i] for i in indices if meta[i]['root'] == root]).mean(axis=0) for root in roots]).mean(axis=0)


def routes(x, checkpoint):
    import numpy as np
    a = x / np.sqrt(np.square(x).mean(axis=-1, keepdims=True) + 1e-6)
    h = a @ checkpoint['ci__proj_in.weight'].T + checkpoint['ci__proj_in.bias']
    h = h / np.sqrt(np.square(h).mean(axis=-1, keepdims=True) + 1e-6)
    return np.minimum(1., np.maximum(0., h @ checkpoint['ci__out_proj.weight'].T + checkpoint['ci__out_proj.bias']))


def matrix_correlations(z, energy):
    import numpy as np
    result = np.full((128, 8), np.nan)
    if len(z) < 3:
        return result
    for feature in range(128):
        x = z[:, feature]
        if x.max() == x.min():
            continue
        for component in range(8):
            y = energy[:, component]
            if y.max() != y.min():
                result[feature, component] = np.corrcoef(x, y)[0, 1]
    return result


def run(backend):
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); verify()
    import numpy as np
    from mechinterp.nanbeige_bpd_gym.probe_towers import load_npz
    base = RUNS / backend; receipt = read(base / 'receipt.json'); inventory = read(ROOT / 'CACHE_INVENTORY.json')['backends'][backend]
    assert receipt['source_manifest_sha256'] == sha(ROOT / 'SOURCE_MANIFEST.json') and receipt['status'] == 'completed'
    results = []
    for cell in receipt['cells']:
        if cell['status'] != 'completed':
            results.append(dict(passes=cell['passes'], status=cell['status'])); continue
        passes = cell['passes']; folder = Path('runs/nan_fit16_io18') / backend / f't{passes}_layer21_down_proj_rank8'
        fit = read(folder / 'receipt.json'); finalfile = folder / fit['final_checkpoint']['file']; initialfile = folder / fit['initial_checkpoint']['file']
        assert sha(finalfile) == cell['checkpoint_sha256']; final, initial = load_npz(finalfile), load_npz(initialfile)
        haarfile = base / f't{passes}_Haar.npz'; assert sha(haarfile) == cell['Haar_sha256']; haar = load_npz(haarfile)
        checked_vectors = 0; checked_codes = 0; checked_correlations = 0
        for data in cell['datasets']:
            dataset = data['dataset']; H, X, Y, meta = load(dataset, backend)
            localfile = base / f't{passes}_{dataset}_local.npz'; assert sha(localfile) == data['local_derivative_sha256']
            with np.load(localfile, allow_pickle=False) as z:
                codes = {layer: z[f'codes_layer{layer}'].copy() for layer in (5, 11, 17, 21)}
                acts = z['component_reads'].copy(); gate = z['controller_masks'].copy()
            np.testing.assert_allclose(acts, X @ final['V'], rtol=1e-10, atol=1e-9)
            np.testing.assert_allclose(gate, routes(X, final), rtol=1e-10, atol=1e-9)
            keys = STRATA[dataset]
            for tower in data['towers']:
                layer = tower['layer_zero_based']; files = inventory['dictionaries'][str(layer)]
                sae = load_npz(files[f'layer{layer}.npz']['path']); random = load_npz(files[f'layer{layer}_random_initial.npz']['path'])
                mean, scale = sae['mean'].astype(float), sae['scale'].astype(float); code = codes[layer]
                pre = ((H[layer]-mean)/scale) @ sae['encoder'].astype(float).T + sae['bias'].astype(float)
                tolerance = 1e-7 * (1 + np.max(np.abs(pre), axis=1))
                assert code.shape == (len(meta), 128) and np.isfinite(code).all() and (code >= 0).all() and ((code > 0).sum(1) <= 4).all()
                # Stable top-k ties are allowed within the frozen coordinate tolerance.
                cutoff = np.sort(pre, axis=1)[:, -4]
                for i in range(len(code)):
                    selected = np.flatnonzero(code[i] > 0)
                    np.testing.assert_allclose(code[i, selected], pre[i, selected], rtol=1e-7, atol=tolerance[i])
                    assert np.all(pre[i, selected] >= cutoff[i]-tolerance[i])
                    assert not np.any((code[i] == 0) & (pre[i] > np.maximum(0, cutoff[i])+tolerance[i]))
                reconstruction = mean + scale * (sae['decoder_bias'].astype(float) + code @ sae['decoder'].astype(float))
                mse = np.square((H[layer]-reconstruction)/scale).mean(1); physical = np.linalg.norm(H[layer]-reconstruction, axis=1)
                rp = ((H[layer]-random['mean'].astype(float))/random['scale'].astype(float)) @ random['encoder'].astype(float).T + random['bias'].astype(float)
                rz = np.zeros_like(rp); selected = np.argsort(-rp, axis=1, kind='stable')[:, :4]
                np.put_along_axis(rz, selected, np.maximum(np.take_along_axis(rp, selected, axis=1), 0), axis=1)
                rr = random['mean'].astype(float) + random['scale'].astype(float) * (random['decoder_bias'].astype(float) + rz @ random['decoder'].astype(float))
                rmse = np.square((H[layer]-rr)/random['scale'].astype(float)).mean(1)
                for group in tower['groups']:
                    ii = [i for i, row in enumerate(meta) if all(row[k] == group[k] for k in keys)]
                    assert group['vectors'] == len(ii) and group['source_root_groups'] == len({meta[i]['root'] for i in ii})
                    for values, field in [(mse, 'normalized_MSE_root_mean'), (rmse, 'random_initial_MSE_root_mean'), (rmse-mse, 'MSE_improvement_root_mean'), (physical, 'physical_reconstruction_L2_root_mean'), ((code > 0).astype(float), 'active_fraction_root_mean')]:
                        np.testing.assert_allclose(macro(values, ii, meta), group[field], rtol=2e-6, atol=2e-7)
                pairs = defaultdict(dict)
                for i, row in enumerate(meta):
                    pairs[row['id']][row['loop']] = i
                for group in tower['loop_gaps']:
                    chosen = [pair for pair in pairs.values() if all(meta[pair[1]][k] == group[k] for k in keys[:-1])]
                    pair_meta = [meta[pair[1]] for pair in chosen]; assert group['contexts'] == len(chosen)
                    for bank, field in [(H[layer], 'physical_displacement_L2_root_mean'), (code, 'sparse_code_displacement_L2_root_mean')]:
                        values = [np.linalg.norm(bank[pair[2]]-bank[pair[1]]) for pair in chosen]
                        np.testing.assert_allclose(macro(np.asarray(values), list(range(len(chosen))), pair_meta), group[field], rtol=1e-10, atol=1e-9)
                checked_codes += code.size
            norms = np.linalg.norm(Y, axis=1); assert (norms > 0).all()
            predictions = [acts @ final['U'], (X @ initial['V']) @ initial['U'], (X @ haar['V']) @ haar['U']]
            errors = [np.linalg.norm(pred-Y, axis=1)/norms for pred in predictions]
            for group in data['learned_groups']:
                ii = [i for i, row in enumerate(meta) if all(row[k] == group[k] for k in keys)]
                for values, field in zip(errors, ('learned_pure_component_relative_L2', 'initial_SVD_relative_L2', 'matched_norm_Haar_relative_L2')):
                    np.testing.assert_allclose(macro(values, ii, meta), group[field], rtol=1e-10, atol=1e-9)
                np.testing.assert_allclose(macro(gate, ii, meta), group['pointwise_gate_root_mean'], rtol=1e-10, atol=1e-9)
                roots = sorted({meta[i]['root'] for i in ii}); loop = group['loop']
                mu = final['mean_comp_acts'][loop-1] if passes == 2 else np.zeros(8)
                energy = np.abs(gate * (acts-mu)) * np.linalg.norm(final['U'], axis=1)
                z = np.asarray([codes[21][[i for i in ii if meta[i]['root'] == root]].mean(0) for root in roots])
                e = np.asarray([energy[[i for i in ii if meta[i]['root'] == root]].mean(0) for root in roots])
                corr = matrix_correlations(z, e)
                got = np.asarray(group['feature_vs_centered_routed_energy_root_correlation'], dtype=float)
                assert got.shape == (128, 8) and np.array_equal(np.isnan(got), np.isnan(corr))
                assert group['undefined_correlation_cells'] == int(np.isnan(corr).sum())
                np.testing.assert_allclose(got, corr, rtol=1e-9, atol=1e-9, equal_nan=True); checked_correlations += got.size
            checked_vectors += len(meta)
        sae = load_npz(inventory['dictionaries']['21']['layer21.npz']['path'])
        E = sae['encoder'].astype(float) / sae['scale'].astype(float)[None, :]; D = sae['decoder'].astype(float) * sae['scale'].astype(float)[None, :]
        for group in cell['geometry']:
            U = final['U'] if group['basis'].startswith('learned_') else initial['U'] if group['basis'].startswith('initial_') else haar['U']
            for rows, field in [(E, 'encoder_vpd_cosine'), (D, 'decoder_vpd_cosine')]:
                denominator = np.linalg.norm(rows, axis=1)[:, None] * np.linalg.norm(U, axis=1)[None, :]
                cosine = np.divide(rows @ U.T, denominator, out=np.zeros_like(denominator), where=denominator > 0)
                np.testing.assert_allclose(cosine, group[field], rtol=1e-10, atol=1e-9)
            np.testing.assert_allclose(E @ U.T, group['encoder_vpd_preactivation_sensitivity'], rtol=1e-10, atol=1e-9)
            assert group['zero_norm_counts'] == dict(encoder_rows=int(np.sum(np.linalg.norm(E, axis=1) == 0)),
                decoder_rows=int(np.sum(np.linalg.norm(D, axis=1) == 0)), VPD_vectors=int(np.sum(np.linalg.norm(U, axis=1) == 0)))
        assert checked_vectors == 1706 and len(cell['geometry']) == 18
        results.append(dict(passes=passes, status='passed', cached_vectors=checked_vectors, sparse_code_values_checked=checked_codes,
            root_correlation_cells_checked=checked_correlations, physical_geometry_cells=18))
    write(RUNS / 'audits' / (backend + '.json'), dict(status='passed', backend=backend, cells=results,
        receipt_sha256=sha(base / 'receipt.json'), source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'),
        limitation='Independent arithmetic from source-locked caches/checkpoints, sharing the cache decoder. No model forward/backward or population inference.', sealed_test_accessed=False))
    print(dict(backend=backend, cells=results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True); a = p.parse_args(); run(a.backend)
