"""Independent NumPy evidence audit; optional checkpoint rerun uses the runner.

Checks archived native output vectors, norm matching, task targets and integer
counts. Source/weight hashes bind artifacts; optimization provenance requires
rerunning the supplied training code.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def require(test,msg):
    if not test:raise AssertionError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(directory):
    r=json.loads((directory/'summary.json').read_text())
    require(sha(directory/'raw.npz')==r['raw_sha256'],'raw digest')
    with np.load(directory/'raw.npz',allow_pickle=False) as z:
        p=z['pairs'];N=len(p);H=r['horizon'];arms=r['arms']
        require(len(set(p[:,0].tolist()))==N,'duplicate item')
        require(np.array_equal(z['current_logits'][:-1],np.broadcast_to(z['receiver_current_logits'],z['current_logits'][:-1].shape)),'current readout altered')
        # Independent graph interpreter validates all oracle labels.
        tokens=z['tokens'];n=r['cfg']['n_nodes'];oracle=z['oracle']
        for b in range(len(tokens)):
            edges=tokens[b,:2*n].reshape(n,2);graph=dict(edges.tolist());current=int(tokens[b,-1])
            for k in range(r['loops']):
                current=graph[current];require(current==int(oracle[k,b]),'wrong graph oracle')
        for i,(b,t,u,label) in enumerate(p):
            require(oracle[t-1,b]==label==oracle[u-1,b],'pair phase mismatch')
            require(z['predictions'][t-1,b]==label==z['predictions'][u-1,b],'endpoint wrong')
            require(np.array_equal(z['expected'][i],oracle[u:u+H,b]),'future oracle mismatch')
        for i in (1,2):require(np.allclose(z['applied_l2'][i],.25*z['donor_edge_l2'],rtol=1e-4,atol=1e-5),'quarter norm')
        for i in (3,4,5):require(np.allclose(z['applied_l2'][i],z['donor_edge_l2'],rtol=1e-4,atol=1e-5),'full norm')
        baseline=z['receiver_logits'].argmax(-1);pred=z['future_logits'].argmax(-1)
        for a,name in enumerate(arms):
            stats=r['summary']['all']['arms'][name]
            require(int((pred[a]!=baseline).any(-1).sum())==stats['future_changed_pairs'],'changed count')
            require(int((pred[a]==z['expected']).sum())==stats['oracle_correct_steps'],'oracle count')
        require(not (pred[0]!=baseline).any(),'sham changed')
        require(not (pred[5]!=baseline).any(),'null changed')
        accuracy=(z['predictions']==oracle).mean(1).tolist()
        require(accuracy==r['oracle_accuracy_by_loop'],'depth accuracy')
    c=json.loads((directory/'calibration.json').read_text())
    require(sha(directory/'calibration.npz')==c['raw_sha256'],'calibration digest')
    with np.load(directory/'calibration.npz',allow_pickle=False) as z:
        require(np.array_equal(z['current_logits'][:-1],np.broadcast_to(z['receiver_current_logits'],z['current_logits'][:-1].shape)),'calibration readout changed')
        baseline=z['receiver_logits'].argmax(-1);pred=z['future_logits'].argmax(-1)
        for a,name in enumerate(arms[:-1]):
            row=c['arms'][name]
            require(row['future_changed_pairs']==int((pred[a]!=baseline).any(-1).sum()),'cross-task changes')
            require(row['next_recipient_agreement']==int((pred[a,:,0]==z['receiver_expected'][:,0]).sum()),'receiver agreement')
            require(row['next_donor_agreement']==int((pred[a,:,0]==z['donor_expected'][:,0]).sum()),'donor agreement')
    return {'seed':r['cfg']['seed'],'pairs':N,'primary_full_edge_changes':r['summary']['all']['arms']['edges_100']['future_changed_pairs'],
            'calibration_full_edge_changes':c['arms']['edges_100']['future_changed_pairs']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
    print(json.dumps([check(x) for x in sorted(a.results.iterdir()) if (x/'summary.json').exists()],indent=2))
