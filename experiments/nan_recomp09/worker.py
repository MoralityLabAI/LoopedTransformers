"""BF16-only execution amendment; source roster and competence screen unchanged."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import torch
from experiments.nan_bpd01.backend import load_backend
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_comp03.prompt import aliases
from experiments.nan_diag04.telemetry import Telemetry
from experiments.nan_gym05.common import prompt_ids
from experiments.nan_fit07.worker import token_record,BudgetReached
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
from mechinterp.nanbeige_bpd_gym.native_components import load_bpd
from mechinterp.nanbeige_bpd_gym.resident_head import mount,retain_decoder_layers,full_logits
from mechinterp.nanbeige_bpd_gym.recomputed_native import rollout_adapter
from .common import read,rows,sha,write,target_digest
from .fit import fit_one


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True)
    p.add_argument('--bpd-root',required=True);p.add_argument('--runs',default='runs/nan_recomp09');a=p.parse_args()
    root=Path(__file__).parent;cfg=read(root/'config.json');seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['worker_files_sha256'].items():assert sha(root/name)==h,name
    for name,h in seal['model_dependencies_sha256'].items():assert sha(name)==h,name
    parent=Path('experiments/nan_fit07');public_file=parent/'fixtures/public.jsonl';baseline_file=Path('runs/nan_fit07/bf16/baseline.json')
    gate_file=Path('runs/nan_fit07/bf16/host_gate.json')
    assert sha(public_file)==seal['public_source_sha256'] and sha(baseline_file)==seal['upstream_baseline_sha256']
    assert sha(gate_file)==seal['upstream_host_gate_sha256']
    gate=read(gate_file);assert gate['allow_fits'] and gate['baseline_sha256']==sha(baseline_file)
    assert sha(Path(a.snapshot)/'tokenizer_config.json')==seal['tokenizer_config_sha256']
    out=Path(a.runs)/'bf16';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();status='failed';failure=None;cell=None
    cache=Path(a.runs)/'HF_cache';cache.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    torch.set_num_threads(1);torch.manual_seed(cfg['source_seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
    def budget():
        if time.monotonic()-started>cfg['maximum_backend_seconds']:raise BudgetReached('Registered resource-amended backend budget reached')
    with Telemetry(out/'memory_samples.json'):
        try:
            assert_oracle_isolated();loop,legacy,tok,_=load_backend(a.compactification_root,a.snapshot,'bf16',chunk_rows=cfg['vocabulary_chunk_rows'])
            retain_decoder_layers(loop,cfg['BF16_resident_decoder_layers'])
            cpu_head=OperatorHead(legacy.weight,aliases(tok),chunk_rows=cfg['vocabulary_chunk_rows']);resident_head=mount(loop,cpu_head);del legacy
            digest=target_digest(loop);original=read(baseline_file)
            assert digest==original['target_parameter_sha256']
            public=rows(public_file);assert len(public)==32
            binding=read(parent/'fixtures/tokenizer_receipt.json');seq={r['id']:prompt_ids(tok,a.snapshot,r['prompt']) for r in public}
            assert all(hashlib.sha256(json.dumps(ids).encode()).hexdigest()==binding['initial_prompt_ids_sha256'][key] for key,ids in seq.items())
            old={r['id']:r for r in original['records']};records=[];parity=[];adapter=rollout_adapter(loop,None,{cfg['site']:cfg['rank']},recompute=False)
            with torch.no_grad():
                for index,row in enumerate(public):
                    budget();logits,boundaries,_,_=adapter(seq[row['id']],2,0,None);phases=[]
                    for t in (0,1):
                        record={'pass_index_zero_based':t,**token_record(logits[t],tok)}
                        assert record==old[row['id']]['phases'][t]
                        if index==0:
                            reference=full_logits(cpu_head.weight,boundaries[t],loop.device,cfg['vocabulary_chunk_rows'])
                            assert torch.equal(reference,logits[t].flatten().cpu())
                            parity.append(dict(loop=t+1,full_vocabulary_logits_checked=len(reference),bit_exact=True))
                        phases.append(record)
                    if index==0:del cpu_head;gc.collect()
                    records.append({k:row[k] for k in ('id','root','split','copy_control')}|{'phases':phases})
                    print({'backend':'bf16','matched_upstream_baseline_contexts':index+1},flush=True)
            write(out/'baseline.json',dict(backend='bf16',records=records,source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),
                upstream_baseline_sha256=sha(baseline_file),public_sha256=sha(public_file),target_parameter_sha256=digest,
                matched_contexts=32,full_head_parity=parity,host_gold_opened=False))
            bank=Path('runs/nan_gym05/bf16/linear_baselines/layer21_down_proj.npz')
            assert sha(bank)==seal['initialization']['bf16']['file_sha256']
            import numpy as np
            with np.load(bank,allow_pickle=False) as z:initial={k:z[k] for k in ('V8','U8')}
            native=loop.model.get_submodule(cfg['site']);effective=native.weight.detach().float().cpu().contiguous().numpy()
            assert hashlib.sha256(memoryview(effective)).hexdigest()==seal['initialization']['bf16']['effective_operator_sha256'];del effective
            source={'BPD_root':a.bpd_root,'manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
                'initial_svd_file_sha256':sha(bank),'effective_operator_sha256':seal['initialization']['bf16']['effective_operator_sha256']}
            budget();cell=fit_one(loop,tok,public,seq,cfg,'bf16',2,out,initial,source,load_bpd(a.bpd_root),budget,digest)
            status='completed' if cell['status']=='completed' else 'completed_with_failed_condition'
            assert_oracle_isolated()
        except Exception as error:
            failure={'type':type(error).__name__,'message':str(error)};traceback.print_exc()
        finally:
            import transformers
            write(out/'receipt.json',dict(backend='bf16',status=status,failure=failure,source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),
                upstream_manifest_sha256=seal['upstream_source_manifest_sha256'],elapsed_seconds=time.monotonic()-started,
                cell_status=cell['status'] if cell else None,optimizer_steps_completed=cell.get('optimizer_steps_completed',0) if cell else 0,
                head_device=str(loop.model.lm_head.weight.device) if 'loop' in locals() else None,
                runtime=dict(torch=torch.__version__,transformers=transformers.__version__,CUDA=torch.version.cuda),
                maximum_added_saved_tensor_CPU_bytes=cfg['maximum_added_saved_tensor_CPU_bytes'],
                formal_inferential_comparisons=0,BPD_donor_transfer_attempted=False,sealed_test_accessed=False))
    if status=='failed':raise SystemExit(2)


if __name__=='__main__':main()
