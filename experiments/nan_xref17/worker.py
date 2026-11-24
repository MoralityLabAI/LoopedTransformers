"""CPU-only complete-cache SAE probes of terminal, independently audited FIT16 cells."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
from .common import ROOT, RUNS, qualified, read, sha, verify, write
from .cache import correlations, groups, load, nullable, root_mean


def loop_gaps(dataset, H, Z, meta):
    import numpy as np
    pairs = defaultdict(dict)
    for i, r in enumerate(meta):
        assert r['loop'] not in pairs[r['id']]
        pairs[r['id']][r['loop']] = i
    values = []; paired_meta = []
    for pair in pairs.values():
        assert set(pair) == {1, 2}; a, b = pair[1], pair[2]
        values.append([float(np.linalg.norm(H[b]-H[a])), float(np.linalg.norm(Z[b]-Z[a]))]); paired_meta.append(meta[a])
    values = np.asarray(values); keys, grouped = groups(dataset, paired_meta, drop_loop=True)
    return [dict(zip(keys, key), contexts=len(ii), source_root_groups=len({paired_meta[i]['root'] for i in ii}),
                 physical_displacement_L2_root_mean=float(root_mean(values[:, 0], ii, paired_meta)),
                 sparse_code_displacement_L2_root_mean=float(root_mean(values[:, 1], ii, paired_meta))) for key, ii in sorted(grouped.items())]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True); a = p.parse_args()
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); seal = verify(); cfg = read(ROOT / 'config.json'); started = time.monotonic()
    import numpy as np
    from mechinterp.nanbeige_bpd_gym.probe_towers import encoding, load_npz
    from mechinterp.nanbeige_bpd_gym.crossref import probe_and_crossref, REVISION
    from experiments.nan_fit07.pointwise import masks
    auditfile = Path('runs/nan_fit16/audits') / (a.backend + '.json')
    audit = read(auditfile) if auditfile.exists() else None
    out = RUNS / a.backend; out.mkdir(parents=True, exist_ok=False)
    inventory = read(ROOT / 'CACHE_INVENTORY.json')['backends'][a.backend]
    for provenance in seal['SAE_tower_manifests'].values():
        assert sha(provenance['path']) == provenance['sha256']
    cells = []; dictionaries = {}
    for layer in cfg['layers_zero_based']:
        names = inventory['dictionaries'][str(layer)]
        for item in names.values():
            assert sha(item['path']) == item['sha256']
        dictionaries[layer] = (load_npz(names[f'layer{layer}.npz']['path']), load_npz(names[f'layer{layer}_random_initial.npz']['path']))
    for passes in cfg['passes']:
        if audit is None or not qualified(audit, passes):
            cells.append(dict(passes=passes, status='unrun_no_passed_terminal_fit_audit')); continue
        folder = Path('runs/nan_fit16') / a.backend / f't{passes}_layer21_down_proj_rank8'
        receipt = read(folder / 'receipt.json'); assert receipt['status'] == 'completed'
        cellaudit = next(c for c in audit['cells'] if c['passes'] == passes)
        assert sha(folder / 'receipt.json') == cellaudit['receipt_sha256']
        finalfile = folder / receipt['final_checkpoint']['file']; initialfile = folder / receipt['initial_checkpoint']['file']
        assert sha(finalfile) == receipt['final_checkpoint']['sha256'] == cellaudit['checkpoint_sha256']
        assert sha(initialfile) == receipt['initial_checkpoint']['sha256']
        final, initial = load_npz(finalfile), load_npz(initialfile); ci = receipt['final_checkpoint']['CI_config']
        replayfile = folder / 'CI_replay_local.npz'; assert sha(replayfile) == receipt['CI_replay_sha256']; replay = load_npz(replayfile)
        replayed = masks(replay['X'], final, ci); np.testing.assert_allclose(replayed, replay['masks'], rtol=2e-5, atol=1e-5)
        rng = np.random.default_rng(cfg['source_seed'] + passes)
        rv = np.linalg.qr(rng.normal(size=final['V'].shape), mode='reduced')[0] * np.linalg.norm(final['V'], axis=0)
        ru = np.linalg.qr(rng.normal(size=final['U'].T.shape), mode='reduced')[0].T * np.linalg.norm(final['U'], axis=1)[:, None]
        controlfile = out / f't{passes}_Haar.npz'
        with controlfile.open('xb') as f:
            np.savez_compressed(f, V=rv, U=ru)
        results = []; geometry = []
        for dataset, expected in cfg['datasets'].items():
            assert time.monotonic() - started < cfg['maximum_host_seconds_per_backend']
            H, X, Y, meta = load(dataset, a.backend); assert len(meta) == expected
            keys, grouped = groups(dataset, meta); codes = {}; tower = []
            for layer in cfg['layers_zero_based']:
                sae, random = dictionaries[layer]
                Z, mse, physical, err = encoding(H[layer], sae, cfg['SAE_top_k'])
                _, random_mse, _, random_err = encoding(H[layer], random, cfg['SAE_top_k']); codes[layer] = Z
                summaries = []
                for key, ii in sorted(grouped.items()):
                    summaries.append(dict(zip(keys, key), vectors=len(ii), source_root_groups=len({meta[i]['root'] for i in ii}),
                        normalized_MSE_root_mean=float(root_mean(mse, ii, meta)), random_initial_MSE_root_mean=float(root_mean(random_mse, ii, meta)),
                        MSE_improvement_root_mean=float(root_mean(random_mse-mse, ii, meta)), physical_reconstruction_L2_root_mean=float(root_mean(physical, ii, meta)),
                        active_fraction_root_mean=root_mean((Z > 0).astype(float), ii, meta).tolist()))
                tower.append(dict(layer_zero_based=layer, physical_coordinate_max_abs=err, random_coordinate_max_abs=random_err,
                    groups=summaries, loop_gaps=loop_gaps(dataset, H[layer], Z, meta)))
            gate = masks(X, final, ci); acts = X @ final['V']
            learned = acts @ final['U']; initial_prediction = (X @ initial['V']) @ initial['U']; random_prediction = (X @ rv) @ ru
            target_norm = np.linalg.norm(Y, axis=1); undefined_denominators = int(np.sum(target_norm == 0))
            relative = lambda prediction: np.divide(np.linalg.norm(prediction-Y, axis=1), target_norm,
                out=np.full(len(Y), np.nan), where=target_norm > 0)
            assert undefined_denominators == 0, 'Registered native nonzero-output coordinate required; preserve undefined failure'
            errors = [relative(pred) for pred in (learned, initial_prediction, random_prediction)]
            readout_groups = []
            for key, ii in sorted(grouped.items()):
                byroot = defaultdict(list)
                for i in ii:
                    byroot[meta[i]['root']].append(i)
                application_loop = meta[ii[0]]['loop']
                mu = final['mean_comp_acts'][application_loop-1] if passes == 2 else np.zeros(8)
                energy = np.abs(gate * (acts-mu)) * np.linalg.norm(final['U'], axis=1)
                root_z = np.asarray([codes[21][v].mean(0) for v in byroot.values()]); root_energy = np.asarray([energy[v].mean(0) for v in byroot.values()])
                corr = correlations(root_z, root_energy, cfg['minimum_defined_correlation_roots'])
                readout_groups.append(dict(zip(keys, key), vectors=len(ii), source_root_groups=len(byroot), out_of_fit_depth=application_loop > passes,
                    learned_pure_component_relative_L2=float(root_mean(errors[0], ii, meta)), initial_SVD_relative_L2=float(root_mean(errors[1], ii, meta)),
                    matched_norm_Haar_relative_L2=float(root_mean(errors[2], ii, meta)), pointwise_gate_root_mean=root_mean(gate, ii, meta).tolist(),
                    feature_vs_centered_routed_energy_root_correlation=nullable(corr), undefined_correlation_cells=int(np.isnan(corr).sum()),
                    scope='Clean cached input application. Centered gated energy is a proxy, not the actual sphere-rescaled online write.'))
            for loop in (1, 2):
                first = next(i for i, r in enumerate(meta) if r['split'] == 'calibration' and r['loop'] == loop
                    and not r.get('copy_control', False) and r.get('arm', 'focused') == 'focused')
                dictionary = inventory['dictionaries']['21']; sae_path = dictionary['layer21.npz']['path']; fit_path = dictionary['layer21_fit_manifest.json']['path']
                site = dict(checkpoint_revision=REVISION, backend=a.backend, layer_zero_based=21, loop=loop, site='residual_after_layer', width=3072)
                for basis, U, digest in [('learned_VPD_T1' if passes == 1 else 'learned_BPD_T2', final['U'], sha(finalfile)),
                                          ('initial_SVD_not_trained', initial['U'], sha(initialfile)), ('matched_norm_Haar_not_trained', ru, sha(controlfile))]:
                    result = probe_and_crossref(H[21][first], dictionaries[21][0], U,
                        state_meta=dict(site, artifact_sha256=meta[first]['state_sha256']),
                        sae_meta=dict(site, artifact_sha256=sha(sae_path), fit_split='calibration', fit_manifest_sha256=sha(fit_path)),
                        vpd_meta=dict(site, artifact_sha256=digest, factor_role='physical_output_directions'))
                    geometry.append(dict(dataset=dataset, application_loop=loop, out_of_fit_depth=loop > passes, basis=basis,
                        bridge='down_proj output identity addition to residual_after_layer before final norm only',
                        **{k: result[k] for k in ('encoder_vpd_cosine', 'decoder_vpd_cosine', 'encoder_vpd_preactivation_sensitivity', 'zero_norm_counts')}))
            localfile = out / f't{passes}_{dataset}_local.npz'
            with localfile.open('xb') as f:
                np.savez_compressed(f, component_reads=acts, controller_masks=gate, **{f'codes_layer{l}': z for l, z in codes.items()})
            write(out / f't{passes}_{dataset}_local_index.json', meta)
            results.append(dict(dataset=dataset, vectors=len(meta), source_root_groups=len({r['root'] for r in meta}), towers=tower,
                learned_groups=readout_groups, local_derivative_sha256=sha(localfile)))
            del H, X, Y, meta, codes, acts, gate, learned, initial_prediction, random_prediction
        cell = dict(passes=passes, status='completed', checkpoint_sha256=sha(finalfile), fit_audit_sha256=sha(auditfile),
            controller_replay_max_abs=float(np.max(np.abs(replayed-replay['masks']))), Haar_sha256=sha(controlfile), datasets=results, geometry=geometry)
        write(out / f't{passes}_aggregate.json', cell); cells.append(cell)
    write(out / 'receipt.json', dict(status='completed', backend=a.backend, cells=cells, source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'),
        elapsed_seconds=time.monotonic()-started, pretrained_model_loaded=False, optimizer_steps=0, formal_inferential_comparisons=0,
        BPD_donor_transfer_attempted=False, sealed_test_accessed=False))
    print(dict(backend=a.backend, cells=[(c['passes'], c['status']) for c in cells]), flush=True)


if __name__ == '__main__':
    main()
