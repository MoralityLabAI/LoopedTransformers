"""Seeded physical norm-matched Haar controls, with no model imports."""
import numpy as np


def control_factors(V,U,seed):
    rng=np.random.default_rng(seed)
    rv=np.linalg.qr(rng.normal(size=V.shape),mode='reduced')[0]*np.linalg.norm(V,axis=0)
    ru=np.linalg.qr(rng.normal(size=U.T.shape),mode='reduced')[0].T*np.linalg.norm(U,axis=1)[:,None]
    return rv.astype(np.float32),ru.astype(np.float32)
