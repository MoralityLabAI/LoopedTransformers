"""One real bank-loading attempt under a separately bounded CPU-only lease."""
from datetime import datetime,timezone
import gc
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).parent
OUT=Path('runs/nan_assets33')


def main():
    from experiments.nan_ram28.common import read,write,sha,THREADS
    seal=read(ROOT/'SOURCE_MANIFEST.json')
    for name,value in seal['critical_files_sha256'].items():assert sha(ROOT/name)==value,name
    for name,value in seal['dependencies_sha256'].items():assert sha(name)==value,name
    cfg=read(ROOT/'registration.json')
    assert sha('experiments/nan_cache32/SOURCE_MANIFEST.json')==cfg['CACHE32_source_manifest_sha256']
    for name in THREADS:os.environ[name]='1'
    os.environ.update(USE_TORCH='0',USE_TF='0',USE_FLAX='0',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    from experiments.nan_ranksite13.resources import below_normal
    from experiments.nan_ram28.resources import memory
    from experiments.nan_storyaudit30 import common as bounded
    below_normal();bounded.no_torch()
    assert not OUT.exists(),'Single attempt only; preserve an earlier result'
    model=bounded.Handle(cfg['model_pid'],int(cfg['model_creation']))
    while memory()['available_physical_bytes']<cfg['minimum_start_available_RAM_bytes']:
        assert model.alive(),'Model ended before bounded asset loading'
        if Path('runs/nan_storyaudit30/audit.log').exists():raise RuntimeError('Yield to registered story audit before asset loading')
        time.sleep(30)
    OUT.mkdir();write(OUT/'admission.json',dict(UTC=datetime.now(timezone.utc).isoformat(),registration=cfg,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json')))
    guard=bounded.Guard(model)
    original_check=guard.check;started=time.monotonic()
    def check():
        assert time.monotonic()-started<=cfg['maximum_loaded_seconds'],'CPU asset loading wall limit'
        assert not Path('runs/nan_storyaudit30/audit.log').exists(),'Yield to registered story audit'
        return original_check()
    guard.check=check;results=[];status='failed';failure=None
    try:
        with patch.object(bounded,'ROOT',ROOT),patch.object(bounded,'RUNS',OUT),guard:
            from importlib.metadata import version
            assert version('numpy')==cfg['numpy_version']
            import numpy as np
            from experiments.nan_cache32.common import verify
            from mechinterp.nanbeige_bpd_gym.chunked_probes import load_fixed_assets,TARGETS
            verify('host');check()
            def metadata(values):
                record={}
                for name,array in sorted(values.items()):
                    check();assert np.isfinite(array).all(),name
                    record[name]=dict(shape=list(array.shape),dtype=str(array.dtype),bytes=array.nbytes)
                return record
            for backend in cfg['backends']:
                check();assets=load_fixed_assets(backend,TARGETS[backend],check)
                assert not assets.MOCK_ONLY and assets.backend==backend
                dictionaries={}
                for layer,pair in assets.dictionaries.items():
                    assert layer in cfg['layers_zero_based'] and len(pair)==2
                    counts=[];records=[]
                    for dictionary in pair:
                        encoder=dictionary['encoder'];features=encoder.shape[0]
                        assert encoder.shape==(features,3072) and features>=4
                        assert dictionary['decoder'].shape==(3072,features)
                        assert dictionary['bias'].shape==(features,)
                        assert dictionary['decoder_bias'].shape==dictionary['mean'].shape==dictionary['scale'].shape==(3072,)
                        assert (dictionary['scale']>0).all()
                        records.append(metadata(dictionary));counts.append(features)
                    assert counts[0]==counts[1];dictionaries[str(layer)]=records
                    del encoder,dictionary,pair
                assert len(dictionaries)==4
                bank=assets.bank
                assert bank['V'].shape==(10752,8) and bank['U'].shape==(8,3072)
                assert bank['mean_comp_acts'].shape==(2,8) and bank['mean_delta_out'].shape==(2,3072)
                assert bank['mean_write_rms'].shape==bank['calibrated_passes'].shape==(2,)
                assert bank['calibrated_passes'].all()
                results.append(dict(backend=backend,status='actual_fixed_assets_loaded',target_binding=assets.target_digest,
                    pins=assets.pins,dictionaries=dictionaries,BPD_bank=metadata(bank),training_updates=assets.asset['receipt']['optimizer_steps_completed']))
                del bank,assets,dictionaries;gc.collect();check()
            status='passed'
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        model.close();bounded.no_torch()
        write(OUT/'receipt.json',dict(status=status,failure=failure,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
            actual_asset_receipts=results,new_model_calls=0,new_fits=0,real_episodes_read=0,task_gold_read=False,
            Torch_imported=False,CUDA_initialized=False,sealed_test_accessed=False,
            evidence='real fixed-asset loading engineering only; no episode, model, CUDA or host-grade admission'))
    print(dict(status=status,backends=len(results),real_episode_reads=0,Torch_imported='torch' in sys.modules),flush=True)


if __name__=='__main__':main()
