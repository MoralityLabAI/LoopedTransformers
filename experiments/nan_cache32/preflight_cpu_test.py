"""GitHub CPU-only production-preflight integration, small RANDOM model.

Fetch exactly three SHA-pinned public code/config files; never fetch weights,
tokenizer data or a task corpus. Use the actual pinned22-layer class, actual
BPD framework, synthetic factors and tiny shapes. CUDA calls/counters are
explicit mocks, not capacity/throughput evidence. Do not run on the user's
RAM-constrained workstation; real model admission remains a separate gate.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import urllib.request

REVISION='451ed48c3273ecef7ea8faaa43c31ce529763bb1'
FILES={'config.json':'f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19',
       'configuration_nanbeige.py':'98abb9faa303717b234b3d381a9650e8cb656ccf287a12f6ad6215a3bbf8be10',
       'modeling_nanbeige.py':'547737f989f5cb741c1a568acaf85f83521fa67a8f7268e19f9a37e60127c0d5'}


def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise RuntimeError('Run this Torch engineering fixture only on GitHub CPU CI')
    p=argparse.ArgumentParser();p.add_argument('--bpd-root',required=True);a=p.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',USE_TORCH='1',USE_TF='0',USE_FLAX='0')
    with tempfile.TemporaryDirectory(prefix='CACHE32_PREFLIGHT_CPU_MOCK_ONLY_') as directory:
        root=Path(directory);snapshot=root/'source';snapshot.mkdir();out=root/'MOCK_ONLY_outputs';out.mkdir()
        for name,expected in FILES.items():
            url=f'https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve/{REVISION}/{name}'
            with urllib.request.urlopen(url,timeout=60) as response:data=response.read(2*1024**2+1)
            assert len(data)<=2*1024**2 and hashlib.sha256(data).hexdigest()==expected
            (snapshot/name).write_bytes(data)
        os.environ.update(HF_HOME=str(root/'HF_home'),HF_MODULES_CACHE=str(root/'HF_modules'))
        import numpy as np
        import torch
        from transformers import AutoConfig,AutoModelForCausalLM
        from experiments.nan_bpd01.backend import NativeLoop
        from experiments.nan_recomp09.common import target_digest
        from mechinterp.nanbeige_bpd_gym.native_components import load_bpd
        from mechinterp.nanbeige_bpd_gym.resident_head import retain_decoder_layers
        from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
        from .common import ROOT,read
        from .worker import Engine
        from .preflight import run
        torch.set_num_threads(1);torch.manual_seed(323232);assert not torch.cuda.is_initialized()
        config=AutoConfig.from_pretrained(snapshot,trust_remote_code=True,local_files_only=True)
        for key,value in dict(hidden_size=64,intermediate_size=128,num_hidden_layers=22,num_attention_heads=8,num_key_value_heads=2,head_dim=16,kv_channels=16,
                vocab_size=256,max_position_embeddings=256,pad_token_id=0,bos_token_id=1,eos_token_id=2,attention_dropout=0.,pretraining_tp=1).items():setattr(config,key,value)
        model=AutoModelForCausalLM.from_config(config,trust_remote_code=True,attn_implementation='sdpa',torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        loop=NativeLoop(model,'cpu');retain_decoder_layers(loop,4);expected=target_digest(loop)
        framework=load_bpd(a.bpd_root);module=importlib.import_module('mechinterp.bpd.ci');site='model.layers.21.mlp.down_proj'
        ci_cfg=framework['config'].CIConfig(variant='state_only',n_blocks=0,d_model=32,mlp_hidden=64,max_passes=2,max_seq_len=32)
        ci=module.TiedCINetwork({site:128},{site:8},cfg=ci_cfg).eval()
        with torch.no_grad():ci.out_proj.bias.fill_(.4)
        rng=np.random.default_rng(3232)
        bank=dict(V=(rng.normal(size=(128,8))*.03).astype(np.float32),U=(rng.normal(size=(8,64))*.03).astype(np.float32),
            mean_comp_acts=np.zeros((2,8),np.float32),mean_delta_out=(rng.normal(size=(2,64))*.02).astype(np.float32),
            mean_write_rms=np.full(2,.07,np.float32),calibrated_passes=np.ones(2,dtype=bool))
        bank.update({'ci__'+name:value.detach().numpy() for name,value in ci.state_dict().items()})
        checkpoint=root/'factors_MOCK_ONLY.npz'
        with checkpoint.open('xb') as stream:np.savez(stream,**bank)
        asset=dict(MOCK_ONLY=True,checkpoint=str(checkpoint),static_CAL_mask=np.asarray([[.25]*8,[.75]*8]),
            receipt=dict(MOCK_ONLY=True,target_parameter_sha256_unchanged=expected,training_contract=dict(ablation=asdict(framework['config'].AblationConfig())),
                         final_checkpoint=dict(CI_config=asdict(ci_cfg))))
        cfg=read(ROOT/'config.json');cfg.update(prefill_token_chunk_rows=3,
            required_capacity=[dict(batch=1,tokens=11,right_padding=0),dict(batch=2,tokens=9,right_padding=1)])
        engine=Engine.__new__(Engine);engine.t=torch;engine.loop=loop;engine.out=out;engine.cfg=cfg
        engine.a=SimpleNamespace(backend='bf16',bpd_root=a.bpd_root);engine.head=StreamedVocabularyHead(model.lm_head.weight,'cpu',8192)
        engine.cache=engine.gym=None;engine.captures=0;engine.preflight_bytes=0
        engine.ids=lambda row:[2,3,5,7,11,13,17];engine.budget=lambda *args:None;engine.observe_peaks=lambda:None
        engine.gpu_check=lambda:dict(MOCK_ONLY=True,CUDA_measurement=False)
        engine.tok=SimpleNamespace(tok=SimpleNamespace(eos_token_id=2,decode=lambda ids,**kwargs:'MOCK_ONLY'))
        import experiments.nan_story23.instrument as instrument
        with (patch.object(instrument,'fit_asset',return_value=asset),
              patch.object(torch.cuda,'empty_cache'),patch.object(torch.cuda,'reset_peak_memory_stats'),
              patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(torch.cuda,'max_memory_reserved',return_value=0)):
            result=run(engine,dict(split='calibration'))
        assert result['status']=='passed' and result['decoder_captures']==17
        assert all(x['status']=='passed' and len(x['passes'])==2 for x in result['capacity'])
        assert not torch.cuda.is_initialized();engine.cache.close();engine.head.close();assert target_digest(loop)==expected
        print(dict(MOCK_ONLY=True,status='actual_production_preflight_small_random_CPU_passed',decoder_captures=engine.captures,
            independent_mask_write_checks=len(result['independent_mask_and_write_checks']),required_mock_shapes=cfg['required_capacity'],
            actual_CUDA_admission=False,pretrained_parameters_loaded=0,pretrained_forwards=0,source_code_config_sha256=FILES,
            largest_mask_error=max(x['maximum_mask_abs_error'] for x in result['independent_mask_and_write_checks']),
            largest_write_relative_L2=max(x['local_write_relative_L2'] for x in result['independent_mask_and_write_checks']),
            real_task_or_gold_read=False,sealed_test_accessed=False),flush=True)


if __name__=='__main__':main()
