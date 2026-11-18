"""Frozen bounded-greedy model-only runner. No oracle/labels imported."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from experiments.nan_bpd01.backend import load_backend
from experiments.nan_bpd01.run import MemorySampler,write,checked_resume
from experiments.nan_comp02.head import OperatorHead
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
from .prompt import PUBLIC_FIELDS,tokens,aliases
from .decode import generate

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['bf16','nf4'],required=True);p.add_argument('--compactification-root',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--runs',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    start=time.perf_counter();runs=Path(a.runs);out=runs/a.backend;out.mkdir(parents=True,exist_ok=True);cohort=runs/'cohort'
    root=Path(__file__).parent;cfg=json.loads((root/'config.json').read_text());frozen=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    for name,h in frozen['critical_files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Frozen source mismatch: '+name)
    for name,h in frozen['reused_files_sha256'].items():
        if sha(root.parents[1]/name)!=h:raise ValueError('Frozen reused source mismatch: '+name)
    fixture=json.loads((runs/'engineering/gpu_fixture.json').read_text())
    if fixture['status']!='passed' or fixture['pretrained_weights_loaded']:raise RuntimeError('Grouped-generation GPU fixture required')
    if sha(runs/'engineering/gpu_fixture.json')!=frozen['gpu_fixture_sha256']:raise RuntimeError('Fixture receipt changed')
    if sha(Path(a.snapshot)/'tokenizer_config.json')!='82316b70213c65e5f820c03d5a11561ccda4055bda984cb770ed0d51fe497c56':raise RuntimeError('Chat pin changed')
    rows={}
    for line in (cohort/'public.jsonl').read_text().splitlines():
        r=json.loads(line)
        if set(r)!=PUBLIC_FIELDS:raise ValueError('Nonpublic model fields')
        rows[r['case_id']]=r
    pairs=json.loads((cohort/'pairs.json').read_text())
    for name,h in frozen['cohort_files_sha256'].items():
        if sha(cohort/name)!=h:raise ValueError('Frozen cohort mismatch: '+name)
    torch.manual_seed(cfg['source_seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
    meta={'backend':a.backend,'public_sha256':sha(cohort/'public.jsonl'),'pairs_sha256':sha(cohort/'pairs.json'),'config_sha256':sha(root/'config.json'),
        'checkpoint_revision':cfg['checkpoint_revision'],'critical_files_sha256':frozen['critical_files_sha256'],
        'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'decoder_source_sha256':sha(root.parent/'nan_bpd01/backend.py'),
        'head_source_sha256':sha(root.parent/'nan_comp02/head.py'),'torch':torch.__version__,'pid':__import__('os').getpid(),
        'quantization':{'format':'bitsandbytes NF4','double_quantization':True,'compute_dtype':'bfloat16'} if a.backend=='nf4' else None,
        'placement':{'embeddings':'CPU','head':'CPU full-vocabulary row chunks','decoder':'one resident BF16 layer plus stage' if a.backend=='bf16' else 'all NF4 decoder CUDA'},
        'oracle_imported':False,'BPD_transfer_attempted':False,'sealed_test_accessed':False,'commands':vars(a),
        'grouped_B1_contexts':6,'KV_cache':False,'maximum_new_tokens':cfg['maximum_new_tokens']}
    prior,completed=checked_resume(out,pairs,meta,a.resume)
    if prior:
        if prior['source_manifest_sha256']!=meta['source_manifest_sha256']:raise ValueError('Resume manifest changed')
        shutil.copyfile(out/'receipt.json',out/'initial_execution_receipt.json')
    status='running';failure=None
    with MemorySampler(out/('memory_samples_resume.json' if prior else 'memory_samples.json')):
        try:
            assert_oracle_isolated();print(f'Loading verified {a.backend} for NAN-COMP-03',flush=True)
            loop,oldhead,tok,entry=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
            alias=aliases(tok)
            if alias!=json.loads((cohort/'source_receipt.json').read_text())['aliases']:raise ValueError('Aliases changed')
            head=OperatorHead(oldhead.weight,alias,chunk_rows=cfg['vocabulary_chunk_rows']);meta['aliases']=alias;meta['eos_token_id']=tok.tok.eos_token_id
            write(out/('manifest_resume.json' if prior else 'manifest.json'),meta)
            for pair in pairs:
                if all(pair[k] in completed for k in ('recipient','donor')):continue
                if time.perf_counter()-start>cfg['maximum_backend_seconds']:status='time_budget_reached';break
                mark=time.perf_counter();print(f'{a.backend} root {pair["root"]} starting',flush=True)
                seq={}
                for role in ('recipient','donor'):
                    for variant in ('curriculum','balanced'):seq[role+'/'+variant]=tokens(tok,a.snapshot,rows[pair[role]],variant)
                seq['erased']=tokens(tok,a.snapshot,rows[pair['recipient']],'erased');seq['erased_repeat']=list(seq['erased'])
                if seq['erased']!=tokens(tok,a.snapshot,rows[pair['donor']],'erased'):raise ValueError('Erasure differs')
                if max(map(len,seq.values()))>cfg['maximum_prompt_tokens']:raise ValueError('Prompt budget')
                records=generate(loop,head,tok.tok,seq,max_tokens=cfg['maximum_new_tokens'])
                for role in ('recipient','donor'):
                    cid=pair[role];rec={'case_id':cid,'root':pair['root'],'split':pair['split'],'family':pair['family'],'role':role,
                        'records':{v:records[role+'/'+v] for v in ('curriculum','balanced')}|{v:records[v] for v in ('erased','erased_repeat')},
                        'erased_inference_reused_within_root':True,'candidate_rows':1}
                    write(out/(cid.replace(':','_')+'.json'),rec);completed.append(cid)
                assert_oracle_isolated();print(f'{a.backend} root {pair["root"]} saved in {time.perf_counter()-mark:.1f}s',flush=True);torch.cuda.empty_cache()
            else:status='completed'
        except Exception as e:
            status='failed';failure={'type':type(e).__name__,'message':str(e)}
            import traceback;traceback.print_exc()
        finally:
            assert_oracle_isolated();torch.cuda.synchronize()
            receipt=meta|{'status':status,'failure':failure,'completed_cases':completed,'elapsed_seconds':round(time.perf_counter()-start,3),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'head_remained_cpu':bool('head' in locals() and head.weight.device.type=='cpu')}
            if prior:
                attempt=dict(receipt);receipt['attempts']=[prior,attempt];receipt['elapsed_seconds']+=prior['elapsed_seconds']
                for k in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes'):receipt[k]=max(prior[k],attempt[k])
            write(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ('status','failure','elapsed_seconds','completed_cases','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes')}),flush=True)
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':main()
