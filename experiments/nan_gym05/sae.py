"""Calibration-only shared-loop sparse bottleneck tower; no language-model edits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .common import read,rows,write,sha

class SmallSAE(torch.nn.Module):
    def __init__(self,width,features,k):
        super().__init__();self.encoder=torch.nn.Linear(width,features);self.decoder=torch.nn.Linear(features,width);self.k=k
        with torch.no_grad():
            torch.nn.init.normal_(self.decoder.weight)
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0,keepdim=True))
            self.encoder.weight.copy_(self.decoder.weight.T);self.encoder.bias.zero_();self.decoder.bias.zero_()
    def encode(self,x):
        a=torch.relu(self.encoder(x));indices=torch.argsort(a,dim=-1,descending=True,stable=True)[...,:self.k]
        values=torch.gather(a,-1,indices)
        return torch.zeros_like(a).scatter(-1,indices,values)
    def forward(self,x):return self.decoder(self.encode(x))

def calibration(folder,layer,expected):
    data=[];provenance=[]
    for file in sorted(folder.glob('root_pointer_*.json')):
        r=read(file)
        if r['split']!='calibration':continue
        if r['root'] not in expected:raise ValueError('Unexpected claimed calibration root')
        statefile=folder/r['state_file'];assert sha(statefile)==r['state_file_sha256']
        with np.load(statefile,allow_pickle=False) as z:
            for o in r['observations']:
                if o['layer_zero_based']==layer and o['site']=='residual_after_layer' and not o['task_id'].endswith(':copy'):
                    if o['task_id'] not in expected[r['root']] or o['loop'] not in (1,2):raise ValueError('Unregistered calibration observation')
                    data.append(z[o['array_key']]);provenance.append({'root':r['root'],'task_id':o['task_id'],'loop':o['loop'],'state_file_sha256':r['state_file_sha256']})
    if len(data)!=256 or len({r['root'] for r in provenance})!=32:raise ValueError('Require all 32 CAL roots, four hops and two native loops')
    actual={(r['task_id'],r['loop']) for r in provenance}
    if actual!={(task,t) for ids in expected.values() for task in ids for t in (1,2)}:raise ValueError('Calibration context coverage mismatch')
    return np.stack(data).astype(np.float32),provenance

def export(model,mean,scale):
    return {'encoder':model.encoder.weight.detach().cpu().numpy(),'bias':model.encoder.bias.detach().cpu().numpy(),
        'decoder':model.decoder.weight.detach().cpu().numpy(),'decoder_bias':model.decoder.bias.detach().cpu().numpy(),'mean':mean,'scale':scale}

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--backend',choices=('bf16','nf4'),required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    root=Path(__file__).parent;cfg=read(root/'config.json');seal=read(root/'SOURCE_MANIFEST.json');folder=Path(a.runs)/a.backend
    for name,h in seal['critical_files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Registered source changed: '+name)
    assert read(folder/'receipt.json')['status']=='completed'
    cohort=Path(a.runs)/'cohort/public.jsonl'
    if sha(cohort)!=seal['combined_public_sha256']:raise ValueError('Calibration source cohort changed')
    expected={}
    for r in rows(cohort):
        if r['split']=='calibration' and r['suite']=='pointer' and not r['id'].endswith(':copy'):
            expected.setdefault(r['root'],set()).add(r['id'])
    out=folder/'sae_tower';out.mkdir(parents=True,exist_ok=True);settings=cfg['SAE'];receipts=[]
    torch.use_deterministic_algorithms(True)
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
    for layer in cfg['layers_zero_based']:
        file=out/f'layer{layer}.npz'
        if file.exists():raise FileExistsError('Never overwrite a fitted tower')
        torch.manual_seed(settings['seed']+layer);X,provenance=calibration(folder,layer,expected)
        if not np.isfinite(X).all():raise ValueError('Nonfinite calibration states')
        mean=X.mean(0);scale=np.maximum(X.std(0),settings['scale_floor']);x=torch.tensor((X-mean)/scale,device=a.device)
        model=SmallSAE(X.shape[1],settings['features'],settings['top_k']).to(a.device)
        initial=export(model,mean,scale);initial={k:v.copy() for k,v in initial.items()}
        initial_file=out/f'layer{layer}_random_initial.npz';np.savez_compressed(initial_file,**initial)
        optimizer=torch.optim.Adam(model.parameters(),lr=settings['lr']);history=[]
        for step in range(settings['steps']):
            idx=torch.randint(len(x),(settings['batch_size'],),device=a.device);batch=x[idx]
            loss=(model(batch)-batch).square().mean()
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite sparse reconstruction loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
            # Unit decoder columns are a training constraint, not an asserted
            # gauge invariance of hard-top-k feature identities.
            with torch.no_grad():
                norms=model.decoder.weight.norm(dim=0).clamp_min(1e-12)
                model.decoder.weight.div_(norms)
            if step%100==0 or step+1==settings['steps']:history.append({'step':step+1,'sample_MSE':float(loss.detach())})
        with torch.no_grad():final_mse=float((model(x)-x).square().mean());codes=model.encode(x);active=(codes>0).any(0)
        if not np.isfinite(final_mse):raise ValueError('Nonfinite final reconstruction')
        np.savez_compressed(file,**export(model,mean,scale))
        fit={'calibration_only':True,'calibration_roots':32,'calibration_vectors':256,'shared_dictionary_for_loops':[1,2],
            'checkpoint_revision':cfg['checkpoint_revision'],'backend':a.backend,'layer_zero_based':layer,'site':cfg['site'],'settings':settings,'provenance':provenance}
        fit['source_manifest_sha256']=sha(root/'SOURCE_MANIFEST.json');fit['combined_public_sha256']=seal['combined_public_sha256']
        fitfile=out/f'layer{layer}_fit_manifest.json';write(fitfile,fit)
        receipt={'file':file.name,'artifact_sha256':sha(file),'fit_manifest_sha256':sha(fitfile),'random_initial_sha256':sha(initial_file),
            'backend':a.backend,'layer_zero_based':layer,'normalized_CAL_MSE':final_mse,'features_ever_active_on_CAL':int(active.sum()),
            'history':history,'DEV_used_for_fitting_or_selection':False,'BPD_transfer_attempted':False}
        write(out/f'layer{layer}_receipt.json',receipt);receipts.append(receipt)
        print(json.dumps({k:receipt[k] for k in ('backend','layer_zero_based','normalized_CAL_MSE','features_ever_active_on_CAL')}),flush=True)
        del x,model,optimizer,codes;torch.cuda.empty_cache()
    write(out/'manifest.json',{'evidence_class':'observational small CAL-trained sparse bottleneck, not a causal BPD instrument','receipts':receipts,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None})

if __name__=='__main__':main()
