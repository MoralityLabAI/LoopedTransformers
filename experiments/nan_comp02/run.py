"""Model-only fresh unary competence run; no oracle import or BPD transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from experiments.nan_bpd01.backend import load_backend
from experiments.nan_bpd01.run import MemorySampler,write,checked_resume
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
from .head import OperatorHead
from .prompt import PUBLIC_FIELDS,tokens,aliases

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['bf16','nf4'],required=True);p.add_argument('--compactification-root',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--runs',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    start=time.perf_counter();runs=Path(a.runs);out=runs/a.backend;out.mkdir(parents=True,exist_ok=True)
    cfgpath=Path(__file__).with_name('config.json');cfg=json.loads(cfgpath.read_text());cohort=runs/'cohort'
    if json.loads((runs/'engineering/gpu_fixture.json').read_text())['status']!='passed':raise RuntimeError('Single-token head GPU gate required')
    if sha(Path(a.snapshot)/'tokenizer_config.json')!='82316b70213c65e5f820c03d5a11561ccda4055bda984cb770ed0d51fe497c56':raise RuntimeError('Chat template pin changed')
    rows={}
    for line in (cohort/'public.jsonl').read_text().splitlines():
        r=json.loads(line)
        if set(r)!=PUBLIC_FIELDS:raise ValueError('Model record contains hidden fields')
        rows[r['case_id']]=r
    pairs=json.loads((cohort/'pairs.json').read_text())
    torch.manual_seed(cfg['source_seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
    meta={'backend':a.backend,'public_sha256':sha(cohort/'public.jsonl'),'pairs_sha256':sha(cohort/'pairs.json'),'config_sha256':sha(cfgpath),
        'checkpoint_revision':cfg['checkpoint_revision'],'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*') if p.suffix in ['.py','.json','.md']},
        'decoder_source_sha256':sha(Path(__file__).parents[1]/'nan_bpd01/backend.py'),'torch':torch.__version__,'pid':__import__('os').getpid(),
        'quantization':{'format':'bitsandbytes NF4','double_quantization':True,'compute_dtype':'bfloat16'} if a.backend=='nf4' else None,
        'placement':{'embeddings':'CPU','head':'CPU with full-vocabulary row chunks','decoder':'one resident BF16 layer plus stage' if a.backend=='bf16' else 'all decoder NF4 CUDA'},
        'checkpoint_verified_before_load':True,'oracle_imported':False,'BPD_transfer_attempted':False,'sealed_test_accessed':False,'commands':vars(a)}
    prior,completed=checked_resume(out,pairs,meta,a.resume)
    if prior and len(completed)==len(rows):print('Already complete');return
    if prior:shutil.copyfile(out/'receipt.json',out/'initial_execution_receipt.json')
    status='running';failure=None
    with MemorySampler(out/('memory_samples_resume.json' if prior else 'memory_samples.json')):
        try:
            assert_oracle_isolated();print(f'Loading verified {a.backend} for NAN-COMP-02',flush=True)
            loop,oldhead,tok,entry=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
            alias=aliases(tok)
            if alias!=json.loads((cohort/'source_receipt.json').read_text())['aliases']:raise ValueError('Answer aliases changed')
            head=OperatorHead(oldhead.weight,alias,chunk_rows=cfg['vocabulary_chunk_rows']);meta['aliases']=alias
            write(out/('manifest_resume.json' if prior else 'manifest.json'),meta)
            def forward(row,variant,both):
                ids=tokens(tok,a.snapshot,row,variant)
                if len(ids)>cfg['maximum_prompt_tokens']:raise ValueError('Prompt limit exceeded')
                e,ctx=loop.prepare(torch.tensor([ids],dtype=torch.long));h1=loop.step(e,ctx,1);h2=loop.step(h1,ctx,2)
                result={'native2':head.score(h2)}
                if both:result['native1']=head.score(h1)
                for record in result.values():record['free_top_token_text']=tok.tok.convert_ids_to_tokens(record['free_top_token_id'])
                return result,len(ids),hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            for pair in pairs:
                if all(pair[r] in completed for r in ['recipient','donor']):continue
                if time.perf_counter()-start>cfg['maximum_backend_seconds']:status='time_budget_reached';break
                mark=time.perf_counter();print(f'{a.backend} root {pair["root"]} starting',flush=True)
                if tokens(tok,a.snapshot,rows[pair['recipient']],'erased')!=tokens(tok,a.snapshot,rows[pair['donor']],'erased'):raise RuntimeError('Erased prefix differs within pair')
                erased,elength,ehash=forward(rows[pair['recipient']],'erased',False)
                repeat,rlength,rhash=forward(rows[pair['recipient']],'erased',False)
                if rhash!=ehash:raise RuntimeError('Repeat-sham inputs changed')
                for role in ['recipient','donor']:
                    cid=pair[role];records={'erased_native2':erased['native2'],'erased_repeat_native2':repeat['native2']};lengths={'erased':elength};hashes={'erased':ehash}
                    for variant in ['direct','worked']:
                        values,length,phash=forward(rows[cid],variant,True);records.update({variant+'_'+k:v for k,v in values.items()});lengths[variant]=length;hashes[variant]=phash
                    rec={'case_id':cid,'root':pair['root'],'split':pair['split'],'family':pair['family'],'role':role,'records':records,
                         'prompt_lengths':lengths,'prompt_token_hashes':hashes,'erased_inference_reused_within_root':True,'candidate_rows':1}
                    write(out/(cid.replace(':','_')+'.json'),rec);completed.append(cid)
                    assert_oracle_isolated()
                print(f'{a.backend} root {pair["root"]} saved in {time.perf_counter()-mark:.1f}s',flush=True);torch.cuda.empty_cache()
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
                attempt=dict(receipt);attempt['completed_cases']=[cid for cid in completed if cid not in prior['completed_cases']]
                receipt['attempts']=[prior,attempt];receipt['elapsed_seconds']+=prior['elapsed_seconds']
                for key in ['peak_cuda_allocated_bytes','peak_cuda_reserved_bytes']:receipt[key]=max(prior[key],attempt[key])
            write(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ['status','failure','elapsed_seconds','completed_cases','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes']}),flush=True)
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':main()
