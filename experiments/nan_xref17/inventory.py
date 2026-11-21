"""Bind existing terminal caches/dictionaries without evaluating vector values."""
from pathlib import Path
from .common import ROOT, read, sha, write


def main():
    inventory = {}
    for backend in ('nf4', 'bf16'):
        groups = {}
        for dataset, experiment, count in [('gym05', 'nan_gym05', None), ('comp14', 'nan_comp14', 240), ('ctx08', 'nan_ctx08', 291)]:
            base = Path('runs') / experiment / backend; receipt = read(base / 'receipt.json'); assert receipt['status'] == 'completed'
            files = sorted(base.glob('root_*.json')) if dataset == 'gym05' else [base / f'case_{i:04d}.json' for i in range(count)]
            entries = []
            for file in files:
                r = read(file); state = base / r['state_file']; assert sha(state) == r['state_file_sha256']
                entries.append(dict(record=str(file).replace('\\', '/'), record_sha256=sha(file), state=str(state).replace('\\', '/'), state_sha256=sha(state)))
            groups[dataset] = dict(receipt=str(base / 'receipt.json').replace('\\', '/'), receipt_sha256=sha(base / 'receipt.json'), files=entries)
        dictionaries = {}
        for layer in (5, 11, 17, 21):
            base = Path('runs/nan_gym05') / backend / 'sae_tower'
            names = [f'layer{layer}.npz', f'layer{layer}_random_initial.npz', f'layer{layer}_fit_manifest.json']
            manifest = read(base / names[-1]); assert manifest['calibration_only'] and manifest['calibration_roots'] == 32 and manifest['calibration_vectors'] == 256
            assert manifest['settings']['steps'] == 1000
            dictionaries[str(layer)] = {name: dict(path=str(base / name).replace('\\', '/'), sha256=sha(base / name)) for name in names}
        groups['dictionaries'] = dictionaries
        groups['independent_gym05_audit'] = dict(path=f'runs/nan_gym05/audit/{backend}_independent.json', sha256=sha(f'runs/nan_gym05/audit/{backend}_independent.json'))
        groups['independent_comp14_audit'] = dict(path=f'runs/nan_comp14/audits/{backend}.json', sha256=sha(f'runs/nan_comp14/audits/{backend}.json'))
        inventory[backend] = groups
    write(ROOT / 'CACHE_INVENTORY.json', dict(backends=inventory, inspection='Closed source filenames, byte hashes and CAL provenance only; no vector analysis, model call or outcome selection.', sealed_test_accessed=False))


if __name__ == '__main__':
    main()
