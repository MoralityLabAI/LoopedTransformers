"""Bounded CPU engineering only: tiny stages plus the pinned meta architecture.

Not a BF16 inference condition or GPU loader. Never opens a real generated
response/state or initializes CUDA. Further BF16 work remains deferred until
its complete loader, resource protocol and CUDA admission are separately frozen.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import threading

MAX_PRIVATE=3*2**28
MIN_AVAILABLE=4*2**30
MIN_START=5*2**30


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args()
    out=Path(a.out_dir).resolve();out.relative_to(Path('runs').resolve())
    assert 'CPU_ONLY' in out.name
    out.mkdir(parents=True,exist_ok=False)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    os.environ.update(USE_TORCH='1',USE_TF='0',USE_FLAX='0',CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',HF_HOME=str(out/'HF_cache'),HF_MODULES_CACHE=str(out/'HF_modules'))
    os.environ.pop('TRANSFORMERS_CACHE',None)
    from experiments.nan_ram28.resources import memory
    from experiments.nan_ranksite13.resources import below_normal
    from experiments.nan_ram28.common import read,write,sha
    below_normal();samples=[];stop=threading.Event()
    def sample():
        m=dict(UTC=datetime.now(timezone.utc).isoformat(),**memory());samples.append(m)
        assert m['private_bytes']<=MAX_PRIVATE and m['available_physical_bytes']>=MIN_AVAILABLE,m
        return m
    assert sample()['available_physical_bytes']>=MIN_START
    write(out/'admission.json',dict(status='admitted_CPU_engineering_only',maximum_private_bytes=MAX_PRIVATE,minimum_available_bytes=MIN_AVAILABLE,
        minimum_start_available_bytes=MIN_START,CPU_threads=1,priority='BelowNormal',pretrained_forwards=0,CUDA_allowed=False))
    def watch():
        while not stop.wait(2):
            try:sample()
            except BaseException as error:
                try:write(out/'resource_failure.json',dict(status='self_only_CPU_engineering_stop',error=str(error),samples=samples))
                finally:os._exit(3)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start();status='failed';failure=None;result={}
    try:
        import torch
        torch.set_num_threads(1);assert not torch.cuda.is_initialized()
        import unittest
        from . import lazy_bf16_storage_tests as tests
        tests.StorageTests.base=out/'synthetic_CPU_ONLY'
        suite=unittest.defaultTestLoader.loadTestsFromModule(tests);ran=unittest.TextTestRunner(verbosity=2).run(suite)
        assert ran.testsRun==8 and not ran.skipped and ran.wasSuccessful(),'CPU storage contracts failed'
        assert not torch.cuda.is_initialized();sample()
        # Authenticate the local checkpoint lock before importing its model code.
        compact=Path(a.compactification_root).resolve();sys.path.insert(0,str(compact))
        os.environ['ARG02_SNAPSHOT_NANBEIGE']=str(Path(a.snapshot).resolve())
        from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
        entry=load_lock()['nanbeige'];assert entry['revision']=='451ed48c3273ecef7ea8faaa43c31ce529763bb1'
        snapshot=Path(a.snapshot)
        for name,digest in entry['files_sha256'].items():assert sha(snapshot/name)==digest,name
        from experiments.nan_ram28.loader import empty_model
        from .lazy_bf16_storage import CheckpointStore,logical_target_digest
        model=empty_model(snapshot);store=CheckpointStore(snapshot);store.validate(model)
        params=list(model.named_parameters());assert len(params)==201 and all(v.is_meta for name,v in params)
        digest=logical_target_digest(model,store)
        expected='cf3db880c052137e7ff7a0dac1e498174f78e32fd3ae9a384aca341fb1d09186'
        assert digest==expected,'Logical ordered tensor bytes differ from the original CTX21 BF16 before-forward target'
        assert all(v.is_meta for name,v in model.named_parameters()) and not torch.cuda.is_initialized();sample()
        result=dict(status='passed_CPU_storage_engineering_only',UTC=datetime.now(timezone.utc).isoformat(),CPU_contracts_passed=8,
            synthetic_two_loop_stage_outputs_bit_exact=True,pinned_meta_parameters=201,
            pinned_checkpoint_parameter_bytes=sum(v.length for v in store.spans.values()),
            logical_target_sha256=digest,reference_condition='Original CTX21 BF16 before-forward parameter manifest',
            logical_bytes_match=True,pretrained_tensor_materialization=False,pretrained_forwards=0,CUDA_initialized=False,
            CPU_threads=1,priority='BelowNormal',full_BF16_condition_registered=False,BF16_inference_status='deferred_host_RAM_policy',
            scope='Storage/hash/meta and tiny CPU-stage engineering only; no pretrained/GPU/RAM/throughput/competence equivalence claim.',
            real_generated_text_or_states_read=False,sealed_test_accessed=False)
        status='passed'
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error));raise
    finally:
        stop.set();watcher.join();sample()
        write(out/'resources.json',dict(status=status,failure=failure,samples=samples,maximum_private_bytes=MAX_PRIVATE,minimum_available_bytes=MIN_AVAILABLE))
    result['resources']=dict(resources_sha256=sha(out/'resources.json'),samples=len(samples),maximum_sampled_private_bytes=max(v['private_bytes'] for v in samples),
        minimum_sampled_available_physical_bytes=min(v['available_physical_bytes'] for v in samples))
    result['source_files_sha256']={name:sha(Path(__file__).with_name(name)) for name in ('lazy_bf16_storage.py','lazy_bf16_storage_tests.py','lazy_bf16_engineering.py')}
    write(out/'engineering_receipt.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
