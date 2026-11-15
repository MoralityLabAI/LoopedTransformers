"""RAM-12R: frozen-trunk historical state-conditioned ranking diagnostics.

Input archives and SHA256 are frozen by experiments/lean_ram12/PROTOCOL.md.
Learns only from source-certified native positive actions and independently
compiler-rejected RAM-08 actions at the SAME exact theorem and tactic prefix.
Does not claim a heldout new-family original Lean theorem proof.
"""
from __future__ import annotations
import argparse,collections,copy,hashlib,importlib.util,io,json,random,re,time,zipfile
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ram08_offline_dev_audit', HERE/'ram08_offline_dev_audit.py')
m=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=m
spec.loader.exec_module(m)

RAM12ZIP='327dbd1382aca1a1c3c790e434341d582aafb85e5ad1b4f9b70ca84f21cae924'
RAM12LABELS='f744f779012c04d50a41798ee3805c91fbb8da922a174b09ccbde3310eb95e8f'
RAM08ZIP=m.ARCHIVE_SHA
COST_WEIGHT=0.2
ALPHA=0.35
UPDATES=150
BATCH=32
SEED=20261115

def sha(raw):return hashlib.sha256(raw).hexdigest()
def family(script):
    if script=='omega':return 'omega'
    if script=='rfl':return 'rfl'
    if script=='decide':return 'decide'
    if script=='simp_all':return 'simp_all'
    if script=='all_goals simp_all':return 'all_simp'
    if script=='assumption':return 'assumption'
    if script=='constructor':return 'constructor'
    if script.startswith('intro ram07_arg'):return 'intro_named'
    if script=='intro':return 'intro'
    if script.startswith('rw ['):return 'rw_local'
    if script.startswith('apply '):return 'apply_local'
    if script.startswith('rcases '):return 'rcases_local'
    if script.startswith('cases '):return 'cases_local'
    if script.startswith('induction '):return 'induction_list'
    if script.startswith('simpa using '):return 'simpa_local'
    if script.startswith('simp only '):return 'simp_only_visible'
    if script.startswith('simp_all only '):return 'simp_all_only_visible'
    if script.startswith('dsimp '):return 'dsimp_only_visible'
    if script.startswith('simp_all [') or script.startswith('all_goals simp_all ['):return 'simp_all'
    if script.startswith('exact '):return 'exact_typed' if script.count(' ')>1 or '(' in script else 'exact_local'
    raise AssertionError('RAM12 historical script outside frozen 20 action families: '+script)

def encode_rows(zip08,zip12,certified):
    if sha(zip08.read_bytes())!=RAM08ZIP or sha(zip12.read_bytes())!=RAM12ZIP:
        raise AssertionError('altered source artifact ZIP input')
    with zipfile.ZipFile(zip08) as z:
        original_raw=z.read('data/transitions.jsonl')
        if sha(original_raw)!=m.DATA_SHA:raise AssertionError('RAM-08 original observations differ')
        old=[json.loads(s) for s in original_raw.decode().splitlines() if s.strip()]
    with zipfile.ZipFile(zip12) as z:
        labels_raw=z.read('certified_state_action_labels.jsonl')
        if sha(labels_raw)!=RAM12LABELS:raise AssertionError('RAM-12 native state labels differ')
        native=[json.loads(s) for s in labels_raw.decode().splitlines() if s.strip()]
    oldmap=collections.defaultdict(list)
    for r in old:
        if not r['development']:oldmap[(r['source_file'],r['theorem'],tuple(r['prefix']))].append(r)
    certmap={(r['root_file'],r['theorem'],tuple(r['prefix_actions']),r['next_action']):r
             for r in [json.loads(s) for s in certified.read_text().splitlines() if s.strip()]}
    if len(certmap)!=230 or len(native)!=230:raise AssertionError('certified source path count differs')
    grouped=collections.defaultdict(list)
    for n in native:
        key=(n['source_file'],n['theorem'],tuple(n['prefix_actions']))
        label=(key[0],key[1],key[2],n['next_action'])
        if label not in certmap:raise AssertionError('native label lacks complete-source historical proof')
        n['remaining_actions']=certmap[label]['min_remaining_actions']
        seen=oldmap[key]
        if seen and seen[0]['before']['goals']!=n['goals']:
            raise AssertionError('exact original theorem/prefix Lean goal mismatch')
        observed=next((r for r in seen if r['script']==n['next_action']),None)
        if observed and observed['label']==0:raise AssertionError('proof-supported action marked original failure')
        fam=observed['family'] if observed else family(n['next_action'])
        grouped[key].append(dict(before={'goals':n['goals']},script=n['next_action'],family=fam,
                                 positive=True,remaining=n['remaining_actions'],split=n['split']))
    dataset=collections.defaultdict(list)
    for key, positives in grouped.items():
        split=positives[0]['split']
        candidates=list(positives)
        for r in oldmap[key]:
            if r['label']!=0 or any(p['script']==r['script'] for p in positives):continue
            candidates.append(dict(before={'goals':positives[0]['before']['goals']},script=r['script'],
                                   family=r['family'],positive=False,remaining=None,split=split))
        dataset[(split,key)]=candidates
    return dataset,old

def features(model,batch):
    d=model.config.width
    fixed=model.fuse(torch.cat([model.mean(batch[k]) for k in ('context','goal','pending','action')]+
                       [model.family(batch['family']),batch['counts']],-1))
    z=fixed
    for _ in range(model.config.loops):z=model.norm(z+model.cell(torch.cat([z,fixed],-1)))
    return z

@torch.no_grad()
def score_rows(model,rows):
    model.eval()
    embeddings=[];rank=[]
    for i in range(0,len(rows),24):
        x=m.encode(rows[i:i+24]);z=features(model,x)
        logits,delta=model(x);p=logits.softmax(-1)
        scores=3*p[:,3]+1.5*p[:,1]+.3*p[:,2]-.6*p[:,0]+.4*delta.clamp(-3,3)
        embeddings.append(z.detach());rank.append(scores.detach())
    return torch.cat(embeddings,0),torch.cat(rank,0)

def precompute(dataset,model):
    states=[];flat=[]
    for (split,key),rows in sorted(dataset.items(),key=lambda x:(x[0][0],x[0][1])):
        offset=len(flat)
        flat+=rows
        states.append(dict(split=split,key=key,indices=list(range(offset,len(flat)))))
    emb,base=score_rows(model,flat)
    return flat,states,emb,base

def pair_indices(states,flat,base):
    success=[];cost=[]
    for st in states:
        if st['split']!='train':continue
        pos=[i for i in st['indices'] if flat[i]['positive']]
        neg=[i for i in st['indices'] if not flat[i]['positive']]
        by_family=collections.defaultdict(list)
        for i in neg:by_family[flat[i]['family']].append(i)
        # One strongest genuine compiler failure per family, selected using the frozen base.
        hard=[max(v,key=lambda i:float(base[i])) for f,v in sorted(by_family.items())]
        for p in pos:
            for n in hard:success.append((p,n))
        for p in pos:
            for q in pos:
                if flat[p]['remaining']<flat[q]['remaining']:
                    cost.append((p,q))
    return success,cost

def metrics(states,flat,scores,kind):
    totals=collections.Counter()
    for st in states:
        if st['split']!=kind:continue
        positives=[i for i in st['indices'] if flat[i]['positive']]
        negatives=[i for i in st['indices'] if not flat[i]['positive']]
        totals['total_prefix_states']+=1
        if positives and negatives:
            totals['states_with_verified_failures']+=1
            pred=max(st['indices'],key=lambda i:(float(scores[i]),-i))
            totals['top1_certified_positive']+=int(pred in positives)
            totals['positive_negative_pairs']+=len(positives)*len(negatives)
            totals['positive_over_failure_pairs']+=sum(float(scores[p])>float(scores[n]) for p in positives for n in negatives)
        if len(positives)>1:
            lengths=[flat[i]['remaining'] for i in positives]
            if len(set(lengths))>1:
                totals['states_with_two_different_certified_suffix_lengths']+=1
                chosen=max(positives,key=lambda i:(float(scores[i]),-i))
                totals['selected_shortest_certified_suffix']+=int(flat[chosen]['remaining']==min(lengths))
                totals['selected_witness_extra_actions']+=flat[chosen]['remaining']-min(lengths)
    result=dict(totals)
    result['positive_over_failure_rate']=result.get('positive_over_failure_pairs',0)/max(1,result.get('positive_negative_pairs',0))
    return result

def run(source08,source12,certified,out):
    if out.exists():raise FileExistsError('preserve existing RAM12 historical experiment output')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    dataset,old=encode_rows(source08,source12,certified)
    out.mkdir(parents=True)
    report=dict(campaign='RAM12R_state_conditioned_bound_residual_posthoc',status='running',
      ram08_sha256=RAM08ZIP,ram12_sha256=RAM12ZIP,labels_sha256=RAM12LABELS,
      train_dev_holdback='source-family hash split /6 fixed in protocol',
      PEF_scores=0,RAM10_scores=0,alpha=ALPHA,updates=UPDATES,batch=BATCH,seed=SEED,
      pair_loss='verified_positive_minus_verifiably_failed_action',
      cost_loss='shorter_vs_longer_both_verified_positive_suffixes',
      unknown_candidates_treated_as_failed=0,model_results={})
    with zipfile.ZipFile(source08) as z:
        for name,(modelseed,digest) in m.EXPECTED.items():
            original=z.read(f'training/{name}_s{modelseed}/best.pt')
            if sha(original)!=digest:raise AssertionError('frozen RAM08 original model SHA mismatch')
            cp=torch.load(io.BytesIO(original),map_location='cpu',weights_only=True)
            model=m.Policy(m.Config(**cp['model_config'])).eval()
            model.load_state_dict(cp['state_dict'])
            model.requires_grad_(False)
            flat,states,emb,base=precompute(dataset,model)
            positive_pairs,cost_pairs=pair_indices(states,flat,base)
            if len(positive_pairs)<50:raise AssertionError('insufficient certified-positive versus original-rejected pairs')
            torch.manual_seed(SEED)
            head=nn.Linear(model.config.width,1)
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
            rng=random.Random(SEED);start=time.perf_counter();losses=[]
            for step in range(1,UPDATES+1):
                selected=[positive_pairs[rng.randrange(len(positive_pairs))] for _ in range(BATCH)]
                pidx=torch.tensor([p for p,n in selected]);nidx=torch.tensor([n for p,n in selected])
                positive_score=base[pidx]+ALPHA*torch.tanh(head(emb[pidx]).squeeze(-1))
                rejected_score=base[nidx]+ALPHA*torch.tanh(head(emb[nidx]).squeeze(-1))
                loss=F.softplus(rejected_score-positive_score).mean()
                if cost_pairs:
                    chosen=[cost_pairs[rng.randrange(len(cost_pairs))] for _ in range(BATCH)]
                    good=torch.tensor([p for p,q in chosen]);longer=torch.tensor([q for p,q in chosen])
                    good_score=base[good]+ALPHA*torch.tanh(head(emb[good]).squeeze(-1))
                    long_score=base[longer]+ALPHA*torch.tanh(head(emb[longer]).squeeze(-1))
                    loss=loss+COST_WEIGHT*F.softplus(long_score-good_score).mean()
                residual=torch.tanh(head(emb[pidx]).squeeze(-1))
                loss=loss+0.02*residual.square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step%50==0:losses.append([step,round(float(loss.detach()),6)])
            with torch.no_grad():
                after=base+ALPHA*torch.tanh(head(emb).squeeze(-1))
                assert bool(((after-base).abs()<=ALPHA+1e-6).all())
            before={k:metrics(states,flat,base,k) for k in ('train','dev','holdback')}
            result={k:metrics(states,flat,after,k) for k in ('train','dev','holdback')}
            save=out/f'{name}_residual_head.pt'
            torch.save(dict(schema='RAM12R-posthoc',base_sha256=digest,alpha=ALPHA,
                         updates=UPDATES,seed=SEED,state_dict=head.state_dict()),save)
            report['model_results'][name]=dict(seed=modelseed,base_sha256=digest,
                residual_sha256=sha(save.read_bytes()),fit_seconds=round(time.perf_counter()-start,3),
                verified_positive_negative_training_pairs=len(positive_pairs),
                shorter_vs_longer_positive_training_pairs=len(cost_pairs),loss_snapshots=losses,
                base=before,updated=result)
            print('RAM12R_MODEL',name,json.dumps(report['model_results'][name],sort_keys=True),flush=True)
    report['status']='complete'
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--ram08',type=Path,required=True)
    a.add_argument('--ram12',type=Path,required=True)
    a.add_argument('--certified-prefixes',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    args=a.parse_args()
    run(args.ram08,args.ram12,args.certified_prefixes,args.out)
