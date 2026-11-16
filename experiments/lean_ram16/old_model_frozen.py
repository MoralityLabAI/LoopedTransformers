"""Read-only reproduction of RAM-08 development-set ranker statistics.

This is not a theorem certification or an evaluation of the frozen PEF test set.
Works against the exact original Actions artifact ZIP, with no network/Lean.
"""
import argparse, collections, dataclasses, hashlib, io, json, re, zipfile
import torch
from torch import nn

FAMILIES=('omega','rfl','decide','simp_all','all_simp','assumption',
 'exact_local','exact_typed','simpa_local','intro','intro_named','constructor',
 'rw_local','apply_local','rcases_local','cases_local','induction_list',
 'simp_only_visible','simp_all_only_visible','dsimp_only_visible')
EXPECTED={
    'loop32':(29,'4a88bc80456743a89a70882911060cc90e76785a5ae1d435a0e46a8f8c09778f'),
    'loop64':(43,'5cbd327904714ac698897e1c2333cae9ec4125aeb70e7c56cd6236db8a737dac'),
    'flat64':(29,'2aa101edbd56f6563ebb2edcfc8c61c07dcc027559ea4cdc551187d6ec5303d8'),
}
ARCHIVE_SHA='a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc'
DATA_SHA='f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f'

@dataclasses.dataclass(frozen=True)
class Config:
    name:str
    width:int
    loops:int

def token_ids(text,length,tail=False):
    tokens=re.findall(r"[\w']+|[^\w\s]",text,re.UNICODE)
    tokens=tokens[-length:] if tail else tokens[:length]
    values=[1+int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'big')%2047 for s in tokens]
    return values+[0]*(length-len(values))

def encode(rows):
    contexts=[];goals=[];pending=[];actions=[];families=[];counts=[]
    for row in rows:
        current=row['before']['goals']
        context,goal=current[0].split('⊢',1)
        contexts.append(token_ids(context,96,True));goals.append(token_ids(goal,96))
        pending.append(token_ids(' || '.join(g.split('⊢',1)[1] for g in current[1:]),128))
        actions.append(token_ids(row['script'],96));families.append(FAMILIES.index(row['family']))
        counts.append([min(8,len(current))/8.,float(len(current)>1)])
    return dict(context=torch.tensor(contexts,dtype=torch.long),goal=torch.tensor(goals,dtype=torch.long),
      pending=torch.tensor(pending,dtype=torch.long),action=torch.tensor(actions,dtype=torch.long),
      family=torch.tensor(families,dtype=torch.long),counts=torch.tensor(counts,dtype=torch.float32))

class Policy(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;d=config.width
        self.embedding=nn.Embedding(2048,d,padding_idx=0);self.family=nn.Embedding(len(FAMILIES),d)
        self.fuse=nn.Sequential(nn.Linear(5*d+2,2*d),nn.GELU(),nn.Linear(2*d,d),nn.GELU())
        if config.loops:
            self.cell=nn.Sequential(nn.Linear(2*d,2*d),nn.GELU(),nn.Linear(2*d,d));self.norm=nn.LayerNorm(d)
        self.head=nn.Linear(d,4);self.delta=nn.Linear(d,1)
    def mean(self,ids):
        mask=(ids!=0).unsqueeze(-1)
        return (self.embedding(ids)*mask).sum(1)/mask.sum(1).clamp_min(1)
    def forward(self,batch):
        fixed=self.fuse(torch.cat([self.mean(batch[k]) for k in ('context','goal','pending','action')]+
                                 [self.family(batch['family']),batch['counts']],-1))
        z=fixed
        for _ in range(self.config.loops):z=self.norm(z+self.cell(torch.cat([z,fixed],-1)))
        return self.head(z),self.delta(z).squeeze(-1)

def goal_shape(state):
    goals=state['goals']
    if not goals:return ('empty',False)
    target=goals[0].split('⊢',1)[-1]
    if '↔' in target:kind='iff'
    elif '→' in target or '∀' in target:kind='implication'
    elif '∧' in target:kind='conjunction'
    elif '=' in target:kind='equality'
    elif any(c in target for c in ('≤','≥','<','>')):kind='inequality'
    else:kind='other'
    return kind,len(goals)>1

def reward(row):return {0:-.6,1:1.5,2:.3,3:3}[row['label']]+.4*row['delta']

def controls(train):
    fgroups=collections.defaultdict(list);sgroups=collections.defaultdict(list)
    for row in train:
        assert not row['development']
        fgroups[(len(row['before']['goals'])>1,row['family'])].append(reward(row))
        sgroups[(goal_shape(row['before']),row['family'])].append(reward(row))
    freq={k:sum(v)/len(v) for k,v in fgroups.items()}
    shaped={k:(sum(v),len(v)) for k,v in sgroups.items()}
    def fn(row):return freq.get((len(row['before']['goals'])>1,row['family']),-.6)
    def sn(row):
        shape=goal_shape(row['before']);baseline=fn(row)
        total,n=shaped.get((shape,row['family']),(0.,0))
        return (total+8*baseline)/(n+8)
    return fn,sn

def count_by_state(rows,scores):
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows):groups[row['state_sha256']].append(i)
    select=[max(ids,key=lambda i:(scores[i],-i)) for ids in groups.values()]
    return dict(states=len(groups),top1_usable=sum(rows[i]['label']>0 for i in select),
                top1_closed=sum(rows[i]['label']==3 for i in select),
                top1_failed=sum(rows[i]['label']==0 for i in select))

def main():
    p=argparse.ArgumentParser();p.add_argument('artifact_zip');p.add_argument('--out',default='/mnt/data/ram08_frozen_dev_audit.json');args=p.parse_args()
    torch.set_num_threads(2)
    raw=open(args.artifact_zip,'rb').read();assert hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA
    z=zipfile.ZipFile(io.BytesIO(raw));data=z.read('data/transitions.jsonl');assert hashlib.sha256(data).hexdigest()==DATA_SHA
    audit=json.loads(z.read('data/independent_audit.json'))
    assert audit['status']=='passed' and audit['checked']==20930 and audit['dataset_sha256']==DATA_SHA
    report=json.loads(z.read('training/training_report.json'));assert report['status']=='complete' and len(report['models'])==9
    rows=[json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]
    train=[r for r in rows if not r['development']];dev=[r for r in rows if r['development']]
    assert len(rows)==20930 and len(train)==18658 and len(dev)==2272
    forbidden={r['state_sha256'] for r in dev};train=[r for r in train if r['state_sha256'] not in forbidden]
    assert len(train)==18658
    f,s=controls(train)
    result=dict(experiment='RAM08_postfreeze_old_source_development_only',
        is_original_PEF_theorem_test=False,PEF_cases_scored=0,artifact_sha256=ARCHIVE_SHA,
        training_rows=len(train),development_rows=len(dev),development_states=len({r['state_sha256'] for r in dev}),
        checkpoint_sha256={},source_data_sha256=DATA_SHA,models={},
        baselines={'native':count_by_state(dev,[-FAMILIES.index(r['family'])-i/1e7 for i,r in enumerate(dev)]),
                   'frequency':count_by_state(dev,[f(r) for r in dev]),
                   'shape_frequency':count_by_state(dev,[s(r) for r in dev])})
    for name,(seed,sha) in EXPECTED.items():
        data=z.read(f'training/{name}_s{seed}/best.pt');assert hashlib.sha256(data).hexdigest()==sha
        cp=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        assert cp['schema']=='RAM08-v1' and cp['seed']==seed and cp['model_config']['name']==name
        assert cp['dataset_sha256']==DATA_SHA
        model=Policy(Config(**cp['model_config'])).eval()
        model.load_state_dict(cp['state_dict'])
        predictions=[]
        with torch.no_grad():
            for i in range(0,len(dev),16):
                logits,delta=model(encode(dev[i:i+16]));p=logits.softmax(-1)
                predictions.extend((3*p[:,3]+1.5*p[:,1]+.3*p[:,2]-.6*p[:,0]+.4*delta.clamp(-3,3)).tolist())
        metrics=count_by_state(dev,predictions)
        registered=report['selected'][name]['dev']
        assert all(metrics[k]==registered[k] for k in ('states','top1_usable','top1_closed')),(name,metrics,registered['top1_usable'],registered['top1_closed'])
        result['models'][name]=dict(seed=seed,parameters=sum(p.numel() for p in model.parameters()),selected_optimizer_step=cp['step'],**metrics)
        result['checkpoint_sha256'][name]=sha
    with open(args.out,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
