"""AF-13: closure-process supervision over the original frozen-source TypedLoop.

Train and development data are the unchanged AF-05 cases. New physical layouts
are reserved entirely for held-out scoring. Initial weights are the two
AF-06 qualified warm-ups, not AF-11's ineligible shared adapter.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper
from experiments.activation_funnel_af11 import experiment as previous
from experiments.activation_funnel_af11.core import replay
from experiments.activation_funnel_af10.core import latch_suffix
from . import data
from .core import trace_upstream,terminal_truth,asynchronous_loss,admissible_status,closure_bounds

SEEDS=(17,29)
ARMS=('terminal','relay_final','async_sound')
STEPS=160;BATCH=32;LR=.0008;WD=.01
PRE={2:4,3:6,4:8,5:10}
FAMILIES=('fresh_depth3','fresh_comp','fresh_depth4','fresh_depth5')
WIDTH=2
PATH=Path(__file__).parent
PROTO=PATH/'PROTOCOL.json'
MANIFEST=PATH/'fresh_manifest.json'
sha=previous.sha_file
write_json=previous.json_write
write_jsonl=previous.jsonl_write
CE=previous.CE


def source_hashes():
    root=Path(__file__).resolve().parents[2]
    deps={
      'activation_funnel_af13':('data.py','core.py','experiment.py','verify.py','aggregate.py','PROTOCOL.json','README.md','fresh_manifest.json','AsyncClosure.lean','tests/test_af13.py'),
      'activation_funnel_af12':('data.py','core.py','experiment.py','fresh_manifest.json','HornClosure.lean'),
      'activation_funnel_af11':('data.py','core.py','experiment.py'),
      'activation_funnel_af10':('core.py',),
      'activation_funnel_af09':('core.py','experiment.py'),
      'activation_funnel_af06':('experiment.py','core.py'),
      'activation_funnel_af05':('gym.py','model.py','experiment.py','corpus_manifest.json'),
      'activation_funnel_af02':('model.py',),
      'activation_funnel':('model.py',),
      'lean_reasoner':('train.py','proof_data.py')}
    return {str(p.relative_to(root)):sha(p)
       for folder,names in deps.items() for n in names
       for p in (root/'experiments'/folder/n,)}


def verify_manifest():
    current=data.fresh_manifest()
    saved=json.loads(MANIFEST.read_text())
    if current!=saved or gym.digest({k:v for k,v in saved.items() if k!='manifest_sha256'})!=saved['manifest_sha256']:
        raise ValueError('AF-13 frozen prospective graph roster mismatch')
    return sha(MANIFEST)


def sampler(seed):
    ds=data.Dataset('train')
    groups={depth:[i for i,c in enumerate(ds.cases) if c.depth==depth] for depth in (2,3)}
    if [len(groups[d]) for d in (2,3)]!=[336,336]:raise ValueError('training corpus drift')
    rng=torch.Generator().manual_seed(202612+seed)
    while True:
        for depth in (2,3):
            ix=torch.randint(len(groups[depth]),(BATCH//2,),generator=rng).tolist()
            indices=[groups[depth][i] for i in ix]
            selected=torch.randint(2,(BATCH//2,),generator=rng)
            rx=ds.get(indices)
            donor,hybrid=ds.pair(indices,selected.tolist())
            yield depth,[ds.cases[i] for i in indices],rx,donor,selected,hybrid


def common_loss(model,recipient,donor,selected,composed,depth):
    pair=replay(model,recipient,donor,selected,PRE[depth],cut=True)
    y=recipient.targets
    c,p,o=pair.clean,pair.patched,pair.control
    loss=(CE(c.goal,y[:,2])+.5*CE(c.initial_q,y[:,:2])+.35*CE(c.q,y[:,:2])
          +.2*CE(o.goal,y[:,2])+.15*CE(o.q,y[:,:2])
          +.75*CE(p.goal,composed[:,2])+.3*CE(p.q,composed[:,:2]))
    return loss,pair


def fit_score(model,ds):
    model.eval(); rows=[]; qs=[]; goals=[]
    with torch.no_grad():
        for depth in (2,3):
            indices=[i for i,c in enumerate(ds.cases) if c.depth==depth]
            for begin in range(0,len(indices),32):
                ids=indices[begin:begin+32];rx=ds.get(ids)
                upstream=model.prefix(rx,PRE[depth]);q,_=model._read(upstream.a,rx.read_slots)
                goal=latch_suffix(model,rx,upstream,3).goal
                qs.extend((q.argmax(-1)==rx.targets[:,:2]).all(-1).tolist())
                goals.extend((goal.argmax(-1)==rx.targets[:,2]).tolist())
                for k,i in enumerate(ids):
                    rows.append({'case_id':ds.cases[i].id,'depth':depth,
                           'q_logits':q[k].tolist(),'goal_logits':goal[k].tolist(),
                           'truth':rx.targets[k].tolist()})
    return {'goal_accuracy':sum(goals)/len(goals),
            'joint_q_accuracy':sum(qs)/len(qs),'rows':len(rows)},rows


def save(out,name,model,seed,receipt,files):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for tensor in state.values():
        if not bool(torch.isfinite(tensor).all()):raise FloatingPointError('nonfinite AF13 checkpoint')
    p=out/(name+'.pt');z=out/(name+'.npz')
    torch.save({'experiment':'AF-13','seed':seed,'arm':name,
                'initial_checkpoint_sha256':receipt['warmup_sha256'],
                'state_dict':state},p)
    np.savez(z,**{k:v.numpy() for k,v in state.items()})
    files[p.name]=sha(p);files[z.name]=sha(z)


def train(archive,out,seed,steps=STEPS,smoke=False):
    if seed not in SEEDS or steps<2 or (not smoke and steps!=STEPS):
        raise ValueError('AF-13 trial identity or fit budget differs from registration')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('AF-13 demands fresh training output')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    manifest_sha=verify_manifest()
    parent_receipt,original,source_archive,warm_sha,train_sha=previous.check_source(archive,seed)
    initial={k:v.detach().clone() for k,v in original.state_dict().items()}
    score_ds={n:data.Dataset(n) for n in ('train','dev')}
    files={};history={};rows=[];fit={}
    for arm in ARMS:
        torch.manual_seed(3413+seed)
        model=GraphLooper(WIDTH);model.load_state_dict(initial)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        rng=sampler(seed);losslog=[];firstgrad=None;start=time.perf_counter()
        for t in range(steps):
            model.train();opt.zero_grad(set_to_none=True)
            sums={}
            for _ in range(2):
                depth,cases,rx,dx,selected,hybrid=next(rng)
                base,_=common_loss(model,rx,dx,selected,hybrid,depth)
                # Every arm sees exactly the same recurrence clock for this update.
                # Extra two calls alternate deterministically over training steps;
                # no model is told the exact Boolean wave arrival timetable.
                horizon=PRE[depth]+(2 if t%2 else 0)
                trace=trace_upstream(model,rx,horizon)
                extra,terms=asynchronous_loss(trace,cases,arm)
                loss=(base+extra)/2
                if not bool(torch.isfinite(loss)):raise FloatingPointError('AF13 nonfinite loss')
                loss.backward()
                for name,val in {'base':float(base.detach()),'total':float((base+extra).detach()),**terms}.items():
                    sums[name]=sums.get(name,0.)+val/2
            if firstgrad is None:
                firstgrad=float(sum(p.grad.square().sum().item() for p in model.cell.parameters() if p.grad is not None)**.5)
                if firstgrad<=0:raise RuntimeError('AF13 recurrent-cell gradient disconnected')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if any(not bool(torch.isfinite(v).all()) for v in model.parameters()):
                raise FloatingPointError('AF13 nonfinite weights')
            if t%40==0 or t==steps-1:losslog.append({'step':t,**sums})
        fit[arm]={};preds=[]
        for split,ds in score_ds.items():
            m,p=fit_score(model,ds);fit[arm][split]=m
            preds.extend([{'arm':arm,'split':split,**x} for x in p])
        rows.extend(preds)
        save(out,arm,model,seed,{'warmup_sha256':warm_sha},files)
        history[arm]={'updates':steps,'seconds':time.perf_counter()-start,
                      'first_cell_grad_l2':firstgrad,'loss_history':losslog}
        print(json.dumps({'seed':seed,'arm':arm,'dev':fit[arm]['dev'],
                          'seconds':history[arm]['seconds']}),flush=True)
    write_jsonl(out/'train_dev_predictions.jsonl',rows)
    files['train_dev_predictions.jsonl']=sha(out/'train_dev_predictions.jsonl')
    receipt={'experiment':'AF-13','seed':seed,'smoke':smoke,'parent_warmup_sha256':warm_sha,
             'parent_training_sha256':train_sha,'parent_zip_sha256':source_archive,
             'source_hashes':source_hashes(),'fresh_manifest_sha256':manifest_sha,
             'protocol_sha256':sha(PROTO),'train_access':['train','dev_eligibility_only'],
             'steps':steps,'arms':list(ARMS),'fit':fit,'train_logs':history,
             'files':files,'original_warmup_fit':parent_receipt['arms']['warmup']['fit_metrics']}
    write_json(out/'train_receipt.json',receipt)
    (out/'train_receipt.sha256').write_text(sha(out/'train_receipt.json')+'\n')
    return receipt


def load_models(folder,seed,smoke=False):
    folder=Path(folder)
    if sha(folder/'train_receipt.json')!=(folder/'train_receipt.sha256').read_text().strip():
        raise ValueError('AF13 training receipt hash mismatch')
    receipt=json.loads((folder/'train_receipt.json').read_text())
    if (receipt['experiment'],receipt['seed'],receipt['smoke'])!=('AF-13',seed,smoke):
        raise ValueError('AF13 trial identity mismatch')
    if receipt['source_hashes']!=source_hashes() or receipt['fresh_manifest_sha256']!=verify_manifest():
        raise ValueError('AF13 source snapshot altered after training')
    if receipt['protocol_sha256']!=sha(PROTO):raise ValueError('AF13 registered protocol mismatch')
    expected={n+'.'+ext for n in ARMS for ext in ('pt','npz')}|{'train_dev_predictions.jsonl'}
    if set(receipt['files'])!=expected:raise ValueError('AF13 checkpoint roster incomplete')
    models={}
    for arm in ARMS:
        p,z=folder/(arm+'.pt'),folder/(arm+'.npz')
        if any(sha(x)!=receipt['files'][x.name] for x in (p,z)):
            raise ValueError('AF13 weight export hash mismatch')
        pkg=torch.load(p,map_location='cpu',weights_only=True)
        if (pkg['experiment'],pkg['seed'],pkg['arm'],pkg['initial_checkpoint_sha256'])!=('AF-13',seed,arm,receipt['parent_warmup_sha256']):
            raise ValueError('AF13 weights provenance changed')
        m=GraphLooper(WIDTH);m.load_state_dict(pkg['state_dict']);m.eval()
        with np.load(z,allow_pickle=False) as values:
            if set(values.files)!=set(pkg['state_dict']) or any(not np.array_equal(values[k],v.numpy()) for k,v in pkg['state_dict'].items()):
                raise ValueError('AF13 named-array weights differ')
        if any(not bool(torch.isfinite(w).all()) for w in m.state_dict().values()):
            raise FloatingPointError('AF13 nonfinite weights')
        models[arm]=m
    if sha(folder/'train_dev_predictions.jsonl')!=receipt['files']['train_dev_predictions.jsonl']:
        raise ValueError('AF13 altered fit predictions')
    return receipt,models


def evaluate(training,out,seed,smoke=False):
    training,out=Path(training),Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('AF13 evaluation requires fresh output')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    receipt,models=load_models(training,seed,smoke)
    splits=('dev',) if smoke else FAMILIES
    summaries={};clean_rows=[];patch_rows=[]
    for arm,model in models.items():
        summaries[arm]={}
        for split in splits:
            ds=data.Dataset(split);totals={'n':len(ds.cases),'joint_q_correct':0,
              'goal_correct':0,'relay_final_correct':0,'relay_final_total':0,
              'unsupported_true_count':0,'missing_fact_count':0,
              'monotonicity_violation_count':0,'fully_admissible_count':0,
              'extended_joint_q_correct':0,'extended_goal_correct':0,'informative':0,'informative_correct':0,
              'unaffected':0,'unaffected_correct':0,'false_goal':0,'patches':0,
              'patch_other_correct':0}
            for begin in range(0,len(ds.cases),16):
                ids=list(range(begin,min(begin+16,len(ds.cases))))
                rx=ds.get(ids);cases=[ds.cases[i] for i in ids]
                depth=cases[0].depth
                if any(c.depth!=depth for c in cases):raise ValueError('depth bucket mixed')
                with torch.no_grad():
                    trace=trace_upstream(model,rx,PRE[depth])
                    prefix_q,_=model._read(trace.final.a,rx.read_slots)
                    clean_goal=latch_suffix(model,rx,trace.final,3).goal
                    extra_trace=trace_upstream(model,rx,PRE[depth]+2)
                    extended_q,_=model._read(extra_trace.final.a,rx.read_slots)
                    extended_goal=latch_suffix(model,rx,extra_trace.final,3).goal
                qok=(prefix_q.argmax(-1)==rx.targets[:,:2]).all(-1)
                gok=clean_goal.argmax(-1)==rx.targets[:,2]
                counts=admissible_status(trace.logits,cases)
                targets=terminal_truth(cases)
                total_atom=(trace.logits[:,-1].argmax(-1)==targets).sum()
                totals['joint_q_correct']+=int(qok.sum());totals['goal_correct']+=int(gok.sum())
                totals['extended_joint_q_correct']+=int((extended_q.argmax(-1)==rx.targets[:,:2]).all(-1).sum())
                totals['extended_goal_correct']+=int((extended_goal.argmax(-1)==rx.targets[:,2]).sum())
                totals['relay_final_correct']+=int(total_atom);totals['relay_final_total']+=targets.numel()
                for k in ('unsupported_true_count','missing_fact_count','monotonicity_violation_count'):
                    totals[k]+=counts[k]
                # This count is recomputed from raw per-stage logits by the independent auditor.
                st=trace.logits.argmax(-1).bool();initial=torch.stack([closure_bounds(c)[0] for c in cases])
                final=targets.bool()
                totals['fully_admissible_count']+=int((~(st & ~final[:,None]).any(-1) & ~(initial[:,None] & ~st).any(-1)).all(-1).sum())
                for row,i in enumerate(ids):
                    clean_rows.append({'arm':arm,'split':split,'case_id':ds.cases[i].id,
                      'depth':depth,'truth':rx.targets[row].tolist(),
                      'prefix_q_logits':prefix_q[row].tolist(),
                      'clean_goal_logits':clean_goal[row].tolist(),
                      'extended_prefix_q_logits':extended_q[row].tolist(),
                      'extended_goal_logits':extended_goal[row].tolist(),
                      'async_stage_logits':trace.logits[row].tolist()})
                for j in (0,1):
                    donor,hybrid=ds.pair(ids,[j]*len(ids))
                    selected=torch.full((len(ids),),j,dtype=torch.long)
                    with torch.no_grad():
                        pair=replay(model,rx,donor,selected,PRE[depth],cut=True)
                    # Exact influence is an oracle property; the model receives no such labels.
                    for row,i in enumerate(ids):
                        c=ds.cases[i];orig=rx.targets[row].tolist();comp=hybrid[row].tolist()
                        inf=orig[1-j] if c.gate=='and' else 1-orig[1-j]
                        pred=int(pair.patched.goal[row].argmax())
                        other=int(pair.patched.q[row,1-j].argmax())
                        totals['informative']+=inf;totals['informative_correct']+=inf*int(pred==comp[2])
                        totals['unaffected']+=1-inf;totals['unaffected_correct']+=(1-inf)*int(pred==comp[2])
                        totals['false_goal']+=int(pred==1 and comp[2]==0);totals['patches']+=1
                        totals['patch_other_correct']+=int(other==orig[1-j])
                        patch_rows.append({'arm':arm,'split':split,'case_id':c.id,'selected':j,
                          'donor_case_id':ds.cases[ds.donors[i,j][0]].id,
                          'recipient':orig,'composed':comp,'influence':inf,
                          'patch_goal_logits':pair.patched.goal[row].tolist(),
                          'patch_q_logits':pair.patched.q[row].tolist(),
                          'untouched_q_code_exact':bool(torch.equal(
                              pair.patched.q_codes[row,:,1-j],pair.patched.q_codes[row,0,1-j].expand_as(pair.patched.q_codes[row,:,1-j])))})
            totals['joint_q_accuracy']=totals['joint_q_correct']/totals['n']
            totals['goal_accuracy']=totals['goal_correct']/totals['n']
            totals['relay_final_accuracy']=totals['relay_final_correct']/totals['relay_final_total']
            totals['extended_joint_q_accuracy']=totals['extended_joint_q_correct']/totals['n']
            totals['extended_goal_accuracy']=totals['extended_goal_correct']/totals['n']
            totals['fully_admissible_fraction']=totals['fully_admissible_count']/totals['n']
            totals['informative_flip_accuracy']=totals['informative_correct']/max(1,totals['informative'])
            totals['irrelevant_retention']=totals['unaffected_correct']/max(1,totals['unaffected'])
            totals['patched_untouched_q_accuracy']=totals['patch_other_correct']/totals['patches']
            summaries[arm][split]=totals
            print(json.dumps({'seed':seed,'arm':arm,'split':split,'q':totals['joint_q_accuracy'],
                          'relay':totals['relay_final_accuracy']}),flush=True)
    write_jsonl(out/'clean.jsonl',clean_rows);write_jsonl(out/'patched.jsonl',patch_rows)
    result={'experiment':'AF-13','seed':seed,'smoke':smoke,
      'train_receipt_sha256':sha(training/'train_receipt.json'),
      'manifest_sha256':receipt['fresh_manifest_sha256'],'source_hashes':receipt['source_hashes'],
      'fit':receipt['fit'],'scores':summaries,'clean_rows':len(clean_rows),'patched_rows':len(patch_rows),
      'clean_sha256':sha(out/'clean.jsonl'),'patched_sha256':sha(out/'patched.jsonl')}
    write_json(out/'evaluation.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--train',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.mode=='train':
        if not a.source_zip:p.error('--source-zip required')
        train(a.source_zip,a.out,a.seed,a.steps,a.smoke)
    else:
        if not a.train:p.error('--train required')
        evaluate(a.train,a.out,a.seed,a.smoke)
if __name__=='__main__':main()
