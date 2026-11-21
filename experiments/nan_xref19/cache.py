"""Load only source-locked, terminal native caches; all vectors retained."""
from collections import defaultdict
from .common import ROOT, read, sha

STRATA = {
    'gym05': ('split', 'suite', 'copy_control', 'loop'),
    'comp14': ('split', 'arm', 'loop'),
    'ctx08': ('split', 'context_budget', 'position', 'task', 'copy_control', 'loop'),
}


def load(dataset, backend):
    import numpy as np
    lock = read(ROOT / 'CACHE_INVENTORY.json')['backends'][backend][dataset]
    assert sha(lock['receipt']) == lock['receipt_sha256']
    H = {l: [] for l in (5, 11, 17, 21)}; X = []; Y = []; meta = []
    for item in lock['files']:
        assert sha(item['record']) == item['record_sha256'] and sha(item['state']) == item['state_sha256']
        r = read(item['record'])
        if dataset == 'gym05':
            by = defaultdict(dict)
            for observation in r['observations']:
                by[observation['task_id'], observation['loop']][observation['layer_zero_based'], observation['site']] = observation['array_key']
        else:
            by = defaultdict(dict)
            for observation in r['state_index']:
                by[r['id'], observation['loop']][observation['layer_zero_based'], observation['site']] = observation['array_key']
        with np.load(item['state'], allow_pickle=False) as z:
            for (key, loop), index in sorted(by.items()):
                row = dict(id=key, root=r['root'], split=r['split'], loop=loop, state_sha256=item['state_sha256'])
                if dataset == 'gym05':
                    row.update(suite=r['suite'], copy_control=key.endswith(':copy'))
                elif dataset == 'comp14':
                    row.update(arm=r['arm'], world=r['world'])
                else:
                    row.update({k: r[k] for k in ('context_budget', 'position', 'task', 'copy_control', 'world')})
                for layer in H:
                    h = z[index[layer, 'residual_after_layer']]; assert h.shape == (3072,) and np.isfinite(h).all()
                    H[layer].append(h)
                x, y = z[index[21, 'down_proj_input']], z[index[21, 'down_proj_output']]
                assert x.shape == (10752,) and y.shape == (3072,) and np.isfinite(x).all() and np.isfinite(y).all()
                X.append(x); Y.append(y); meta.append(row)
    expected = read(ROOT / 'config.json')['datasets'][dataset]
    assert len(meta) == expected and len({(r['id'], r['loop']) for r in meta}) == expected
    return {l: np.asarray(v, dtype=np.float64) for l, v in H.items()}, np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), meta


def groups(dataset, meta, drop_loop=False):
    result = defaultdict(list); keys = STRATA[dataset][:-1] if drop_loop else STRATA[dataset]
    for i, r in enumerate(meta):
        result[tuple(r[k] for k in keys)].append(i)
    return keys, result


def root_mean(values, indices, meta):
    import numpy as np
    byroot = defaultdict(list)
    for i in indices:
        byroot[meta[i]['root']].append(i)
    return np.mean([np.mean(values[v], axis=0) for v in byroot.values()], axis=0)


def correlations(a, b, minimum_roots=3):
    import numpy as np
    assert a.ndim == b.ndim == 2 and len(a) == len(b)
    if len(a) < minimum_roots:
        return np.full((a.shape[1], b.shape[1]), np.nan)
    variable_a = np.ptp(a, axis=0) > 0; variable_b = np.ptp(b, axis=0) > 0
    a = a - a.mean(0); b = b - b.mean(0)
    denominator = np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)[None, :]
    return np.divide(a.T @ b, denominator, out=np.full(denominator.shape, np.nan),
        where=(denominator > 0) & variable_a[:, None] & variable_b[None, :])


def nullable(value):
    import numpy as np
    return [[float(v) if np.isfinite(v) else None for v in row] for row in value]
