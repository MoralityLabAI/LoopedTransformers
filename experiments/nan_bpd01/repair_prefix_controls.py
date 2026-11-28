"""Post-readout software repair, fixed cases/seeds/dose; no source reselection."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .backend import load_backend,candidate_inputs
from .edits import shared_random_dark,dose_edit,prefix_hidden_error
from .head import prefix_logprob_spread
from .run import MemorySampler,write
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated,public_rows

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['bf16','nf4'],required=True);p.add_argument('--runs',required=True)
    p.add_argument('--compactification-root',required=True);p.add_argument('--snapshot',required=True);a=p.parse_args()
    start=time.perf_counter();runs=Path(a.runs);out=runs/'prefix_repair'/a.backend
    if (out/'receipt.json').exists():raise FileExistsError('Repair receipts are immutable; choose a separately audited attempt')
    out.mkdir(parents=True,exist_ok=True);cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    torch.manual_seed(cfg['seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
    rows={r['case_id']:r for r in public_rows(runs/'cohort/public.jsonl')};pairs=json.loads((runs/'cohort/pairs.json').read_text())
    selected=[p for p in pairs if p['split']=='development' and cfg['development_roots'].index(p['root']) in cfg['phase_intervention_root_indices']]
    meta={'backend':a.backend,'scope':'post-readout prefix-consistency software repair; exploratory; same fixed cases, seeds and dose',
        'public_sha256':hashlib.sha256((runs/'cohort/public.jsonl').read_bytes()).hexdigest(),'checkpoint_revision':cfg['checkpoint_revision'],
        'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['repair_prefix_controls.py','backend.py','head.py','edits.py']},
        'original_outcomes_exposed_before_repair':True,'new_sources_selected':False,'sealed_test_accessed':False,'commands':vars(a)}
    completed=[];status='running';failure=None
    with MemorySampler(out/'memory_samples.json'):
        try:
            print(f'Loading {a.backend} for two fixed prefix-control repairs',flush=True)
            loop,head,tok,entry=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
            for pair in selected:
                if time.perf_counter()-start>cfg['maximum_backend_seconds']:raise RuntimeError('Bound reached before repair pair completion')
                cid=pair['recipient'];print(f'{a.backend} repair root {pair["root"]} starting',flush=True)
                ids,_=candidate_inputs(tok,dict(zip('INSR',head.ids)),a.snapshot,rows[cid]);emb,ctx=loop.prepare(ids)
                h1=loop.step(emb,ctx,1);h2=loop.step(h1,ctx,2);native=head.score([h1,h2])
                original=json.loads((runs/a.backend/(cid.replace(':','_')+'.json')).read_text());parity=[]
                for name,r in zip(['native1','native2'],native):
                    parity.append(max(abs(x-y) for x,y in zip(r['scores'],original['records'][name]['scores'])))
                if max(parity)!=0:raise RuntimeError(f'Native score repeat failed exact parity: {parity}')
                if max(prefix_hidden_error(h1),prefix_hidden_error(h2))!=0:raise RuntimeError('Native candidate prefix closure failed')
                record={'case_id':cid,'root':pair['root'],'native_score_repeat_max_errors':parity,'interventions':{}}
                for seed in [741,947]:
                    name=f'random_shared_dark_{seed}'
                    direction=shared_random_dark(h1,width=cfg['context_width'],seed=pair['root']+seed)
                    edited,dose=dose_edit(h1,direction,cfg['physical_relative_dose'],cfg['maximum_relative_dose_error'])
                    current=head.score([h1,edited])[1];nxt=loop.step(edited,ctx,2);delayed=head.score([h2,nxt])[1]
                    hidden_error=max(prefix_hidden_error(edited),prefix_hidden_error(nxt));spread=max(prefix_logprob_spread(current),prefix_logprob_spread(delayed))
                    if hidden_error!=0 or spread!=0:raise RuntimeError(f'Shared prefix repair failed: {hidden_error}, {spread}')
                    if current['full_vocab_max_logprob_error_vs_reference']!=0:raise RuntimeError('Current readout parity failed')
                    record['interventions'][name]={'dose':dose,'current':current,'delayed':delayed,'next_state_l2':float((nxt.float()-h2.float()).norm()),
                        'cosine_to_donor_next_displacement':None,'prefix_hidden_max_error':hidden_error,'prefix_logprob_spread':spread,
                        'noise_rule':'first one-row CUDA Gaussian draw with original seed, broadcast over all 16 shared-prefix candidates'}
                    del direction,edited,nxt
                write(out/(cid.replace(':','_')+'.json'),record);completed.append(cid);print(f'{a.backend} repair root {pair["root"]} saved',flush=True)
                del h1,h2,emb;torch.cuda.empty_cache()
            status='completed'
        except Exception as e:
            status='failed';failure={'type':type(e).__name__,'message':str(e)}
            import traceback;traceback.print_exc()
        finally:
            assert_oracle_isolated();torch.cuda.synchronize()
            receipt=meta|{'status':status,'failure':failure,'completed_cases':completed,'elapsed_seconds':round(time.perf_counter()-start,3),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'head_remained_cpu':bool('head' in locals() and head.weight.device.type=='cpu')}
            write(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ['status','failure','elapsed_seconds','completed_cases']}),flush=True)
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':main()
