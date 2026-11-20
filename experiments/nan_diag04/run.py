"""Frozen model-only component diagnostic. All gold assistance is explicit in prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from experiments.nan_bpd01.backend import load_backend
from experiments.nan_bpd01.run import write
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_comp02.prompt import aliases
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
from .prompt import tokens
from .decode import generate
from .telemetry import Telemetry

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('bf16','nf4'),required=True);p.add_argument('--compactification-root',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--runs',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();start=time.perf_counter();root=Path(__file__).parent;runs=Path(a.runs);out=runs/a.backend;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'config.json').read_text());frozen=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    if sha(Path(a.snapshot)/'tokenizer_config.json')!='82316b70213c65e5f820c03d5a11561ccda4055bda984cb770ed0d51fe497c56':raise ValueError('Pinned tokenizer template changed')
    for name,h in frozen['critical_files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Frozen source changed: '+name)
    for name,h in frozen['reused_files_sha256'].items():
        if sha(root.parents[1]/name)!=h:raise ValueError('Reused source changed: '+name)
    for name,h in frozen['model_input_files_sha256'].items():
        if sha(runs/'cohort'/name)!=h:raise ValueError('Frozen public source changed')
    if sha(runs/'engineering/gpu_fixture.json')!=frozen['gpu_fixture_sha256']:raise ValueError('Fixture seal changed')
    public={r['root']:r for r in map(json.loads,(runs/'cohort/public.jsonl').read_text().splitlines())}
    assert all(set(r)=={'root','split','family','prompts','public_sha256'} for r in public.values())
    meta={'backend':a.backend,'checkpoint_revision':cfg['checkpoint_revision'],'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'critical_files_sha256':frozen['critical_files_sha256'],
        'public_sha256':sha(runs/'cohort/public.jsonl'),'config_sha256':sha(root/'config.json'),'pid':__import__('os').getpid(),'torch':torch.__version__,'commands':vars(a),
        'placement':{'head':'CPU BF16 full-vocabulary chunks','embeddings':'CPU BF16','decoder':'one BF16 resident layer plus stage' if a.backend=='bf16' else 'CUDA NF4 double-quant BF16 compute'},
        'grouped_independent_B1_contexts':24,'KV_cache':False,'oracle_imported':False,'BPD_transfer_attempted':False,'sealed_test_accessed':False}
    prior=None;completed=[]
    if (out/'receipt.json').exists():
        if not a.resume:raise FileExistsError('Use new runs or one explicit resume')
        prior=json.loads((out/'receipt.json').read_text());assert not prior.get('attempts')
        for k in ('source_manifest_sha256','backend','public_sha256','config_sha256'):assert prior[k]==meta[k]
        completed=list(prior['completed_roots']);assert set(completed)<=set(public)
        shutil.copyfile(out/'receipt.json',out/'initial_execution_receipt.json')
        for k in completed:assert json.loads((out/f'root_{k}.json').read_text())['root']==k
    elif a.resume:raise FileNotFoundError('No resume receipt')
    torch.manual_seed(cfg['seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats();status='running';failure=None
    with Telemetry(out/('memory_samples_resume.json' if prior else 'memory_samples.json')):
        try:
            assert_oracle_isolated();print(f'Loading verified {a.backend} for NAN-DIAG-04',flush=True)
            loop,oldhead,tok,entry=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows']);head=OperatorHead(oldhead.weight,aliases(tok),chunk_rows=cfg['vocabulary_chunk_rows'])
            meta['aliases']=aliases(tok);meta['eos_token_id']=tok.tok.eos_token_id;write(out/('manifest_resume.json' if prior else 'manifest.json'),meta)
            for k,r in public.items():
                if k in completed:continue
                if time.perf_counter()-start>cfg['maximum_backend_seconds']:status='time_budget_reached';break
                mark=time.perf_counter();print(f'{a.backend} root {k} starting',flush=True)
                seq={n:tokens(tok,a.snapshot,v) for n,v in r['prompts'].items()};assert len(seq)==24 and max(map(len,seq.values()))<=cfg['maximum_prompt_tokens']
                assert seq['erased']==seq['erased_repeat']
                records=generate(loop,head,tok.tok,seq,{n:v['kind'] for n,v in r['prompts'].items()},cfg)
                write(out/f'root_{k}.json',{'root':k,'split':r['split'],'family':r['family'],'records':records});completed.append(k)
                assert_oracle_isolated();print(f'{a.backend} root {k} saved in {time.perf_counter()-mark:.1f}s',flush=True);torch.cuda.empty_cache()
            else:status='completed'
        except Exception as e:
            status='failed';failure={'type':type(e).__name__,'message':str(e)}
            import traceback;traceback.print_exc()
        finally:
            assert_oracle_isolated();torch.cuda.synchronize();receipt=meta|{'status':status,'failure':failure,'completed_roots':completed,'elapsed_seconds':round(time.perf_counter()-start,3),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'head_remained_cpu':bool('head' in locals() and head.weight.device.type=='cpu')}
            if prior:
                receipt['attempts']=[prior,dict(receipt)];receipt['elapsed_seconds']+=prior['elapsed_seconds']
                for name in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes'):receipt[name]=max(prior[name],receipt[name])
            write(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ('status','failure','completed_roots','elapsed_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes')}),flush=True)
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':main()
