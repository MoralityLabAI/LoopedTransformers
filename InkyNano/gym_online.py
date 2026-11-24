"""Incremental, cached native teacher routing pilot.

Full-batch native routing was not prefix-invariant on the dev root. This
separate online instrument computes each token before its future is known;
never relabel the full-batch traces as causal. Outputs remain pilot data.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from inventory import MODEL,REV
from gym_export import modules

TRANSFORMERS_REV='d8f10a82213a3a9db48385aba3a4b57ab535de50'

def dump(path, obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

@torch.inference_mode()
def capture_online(model,routers,prompt,continuation=None,max_new_tokens=160,deadline=None):
    """Capture a prompt then feed each forced or greedy token through the cache.

    A generated token is processed explicitly to capture its router visits;
    generation's last predicted token is never silently omitted.
    """
    events={l:[] for l in routers};handles=[];cache=None;ids=list(prompt)
    chunks=[]
    def make(layer):
        def hook(_module,_inputs,out):
            _,w,ix=out
            events[layer].append((ix.detach(),w.detach()))
        return hook
    try:
        for l,m in routers.items():handles.append(m.register_forward_hook(make(l)))
        def step(tokens):
            nonlocal cache
            if deadline and time.monotonic()>deadline-45:raise TimeoutError('routing deadline')
            before={l:len(events[l]) for l in routers}
            t=torch.tensor([tokens],device='cuda:0',dtype=torch.long)
            out=model(input_ids=t,attention_mask=torch.ones((1,len(ids)),device='cuda:0',dtype=torch.long),
                      cache=cache,use_cache=True,logits_to_keep=1)
            cache=out.cache
            if cache is None:raise ValueError('native cache missing')
            for l in routers:
                n=3 if 6<=l<18 else 1
                new=events[l][before[l]:]
                if len(new)!=n or any(ix.shape!=(len(tokens),8) or w.shape!=ix.shape for ix,w in new):
                    raise ValueError(f'missing native visits at layer {l}')
            return out.logits[0,-1]
        logits=step(prompt)
        if continuation is not None:
            for token in continuation:
                ids.append(int(token));logits=step([int(token)])
        else:
            for _ in range(max_new_tokens):
                next_token=int(torch.argmax(logits).item())
                ids.append(next_token);logits=step([next_token])
                eos=model.config.get_text_config().eos_token_id
                if next_token in (eos if isinstance(eos,(list,tuple)) else [eos]):break
    finally:
        for h in handles:h.remove()
    visits={}
    for l,outputs in events.items():
        n=3 if 6<=l<18 else 1
        per=[]
        for pass_id in range(n):
            parts=outputs[pass_id::n]
            ix=torch.cat([a for a,_ in parts],dim=0)
            w=torch.cat([b for _,b in parts],dim=0)
            if ix.shape!=(len(ids),8):raise ValueError('incomplete token/visit alignment')
            per.append((ix.cpu(),w.cpu()))
        visits[l]=per
    return ids,visits


def same_prefix(a,b,n):
    mismatch=0;max_delta=0.;total=0
    for l in a:
        for (ix,w),(ix2,w2) in zip(a[l],b[l]):
            i,j=ix[:n].argsort(-1),ix2[:n].argsort(-1)
            mismatch+=int((ix[:n].gather(-1,i)!=ix2[:n].gather(-1,j)).any(-1).sum())
            max_delta=max(max_delta,float((w[:n].gather(-1,i)-w2[:n].gather(-1,j)).abs().max()))
            total+=n
    return {'id_mismatch_rows':mismatch,'compared_rows':total,'max_abs_weight_diff':max_delta}


def write_roster_and_visits(r,arm,prompt_len,ids,vs,rf,tf,continuation,finish_reason=None):
    rf.write(json.dumps({'root':r['root'],'turn':arm,'family':r['family'],'split':r['split'],
        'source_kind':r['source_kind'],'source':r['source'],'category':r['category'],
        'arm':arm,'token_ids':ids,'prompt_len':prompt_len,
        'prompt':r['prompt'],'continuation':continuation,'finish_reason':finish_reason})+'\n')
    for l,per in vs.items():
        for pass_id,(ix,w) in enumerate(per):
            for t,(selected,weight) in enumerate(zip(ix.tolist(),w.tolist())):
                z=sum(weight)
                tf.write(json.dumps({'root':r['root'],'turn':arm,'token':t,'layer':l,'pass':pass_id,
                    'experts':selected,'weights':weight,
                    'normalized_weights':[v/z for v in weight],
                    'phase':'prompt' if t<prompt_len else 'continuation','arm':arm})+'\n')
    rf.flush();tf.flush()

@torch.inference_mode()
def main():
    a=argparse.ArgumentParser();a.add_argument('--snapshot',type=Path,required=True)
    a.add_argument('--runtime-receipt',type=Path,required=True)
    a.add_argument('--roster',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--limit-seconds',type=int,default=1800)
    a.add_argument('--max-new-tokens',type=int,default=160);a.add_argument('--expected-roots',type=int,default=16);args=a.parse_args()
    if json.loads(args.runtime_receipt.read_text())['commit']!=TRANSFORMERS_REV:
        raise ValueError('wrong Transformers revision')
    import transformers
    from transformers import AutoTokenizer,Inkling2LoopForConditionalGeneration
    if transformers.__version__!='5.24.0.dev0' or torch.cuda.device_count()!=1:raise ValueError('runtime mismatch')
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+args.limit_seconds
    roster=[json.loads(s) for s in args.roster.read_text().splitlines() if s.strip()]
    if not 1<=args.expected_roots<=16 or len(roster)!=args.expected_roots or len({r['root'] for r in roster})!=len(roster) or len({r['family'] for r in roster})!=len(roster):
        raise ValueError('each shard requires its declared independent roots (max 16)')
    if any(r['split'] not in ('train','dev') or r['source_kind'] not in ('public','synthetic') for r in roster):
        raise ValueError('only public/synthetic train/dev roots')
    if not any(r['split']=='dev' for r in roster):raise ValueError('need dev root for prefix check')
    state={'source_model':MODEL,'source_revision':REV,'transformers_revision':TRANSFORMERS_REV,
           'instrument':'cached incremental native forward, separate from full batch',
           'status':'loading','arms':{},'start_roster_sha256':hashlib.sha256(args.roster.read_bytes()).hexdigest()}
    dump(out/'online_receipt.json',state)
    model=Inkling2LoopForConditionalGeneration.from_pretrained(str(args.snapshot),dtype=torch.bfloat16,
        device_map={'':'cuda:0'}).eval()
    tok=AutoTokenizer.from_pretrained(str(args.snapshot));routers=modules(model)['gate']
    state['load_seconds']=time.monotonic()-start
    dev=next(r for r in roster if r['split']=='dev')
    p=tok.encode(dev['prompt'],add_special_tokens=True)
    c=tok.encode(dev['continuation'],add_special_tokens=False)
    if len(c)<2:raise ValueError('prefix diagnostic needs two forced tokens')
    short_ids,short=capture_online(model,routers,p,c[:2],deadline=deadline)
    alt=tok.encode(' independent suffix',add_special_tokens=False)[0]
    long_ids,long=capture_online(model,routers,p,c[:2]+[alt],deadline=deadline)
    check=same_prefix(short,long,len(short_ids))
    state['online_prefix_check']=check
    dump(out/'online_prefix_check.json',check);dump(out/'online_receipt.json',state)
    if check['id_mismatch_rows'] or check['max_abs_weight_diff']>1e-4:
        state['status']='blocked_online_prefix_invariance';dump(out/'online_receipt.json',state)
        raise ValueError('cached online prefix mismatch')
    state['status']='capturing'
    for arm in ('reference_forced','generated'):
        d=out/arm;d.mkdir(exist_ok=True)
        n=0;lengths=[]
        with (d/'roster.jsonl').open('w') as rf,(d/'trace.jsonl').open('w') as tf:
            for r in roster:
                if time.monotonic()>deadline-150:break
                p=tok.encode(r['prompt'],add_special_tokens=True)
                if arm=='reference_forced':
                    c=tok.encode(r['continuation'],add_special_tokens=False)
                    ids,vs=capture_online(model,routers,p,c,deadline=deadline)
                    text=r['continuation']
                else:
                    ids,vs=capture_online(model,routers,p,None,args.max_new_tokens,deadline=deadline)
                    text=tok.decode(ids[len(p):],skip_special_tokens=False)
                eos=model.config.get_text_config().eos_token_id
                eos_ids=eos if isinstance(eos,(list,tuple)) else [eos]
                reason='reference' if arm=='reference_forced' else ('eos' if ids[-1] in eos_ids else 'token_cap')
                write_roster_and_visits(r,arm,len(p),ids,vs,rf,tf,text,reason)
                n+=1;lengths.append(len(ids)-len(p))
                state['arms'][arm]={'roots':n,'continuation_lengths':lengths,'complete':False}
                dump(out/'online_receipt.json',state)
        state['arms'][arm]={'roots':n,'continuation_lengths':lengths,'complete':n==len(roster)}
        dump(out/'online_receipt.json',state)
    state['elapsed_seconds']=time.monotonic()-start
    state['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    state['status']='complete' if all(v['complete'] for v in state['arms'].values()) else 'partial'
    dump(out/'online_receipt.json',state)
    print(json.dumps({'status':state['status'],'arms':state['arms'],
        'elapsed_seconds':state['elapsed_seconds']}))
if __name__=='__main__':main()
