"""Freeze model/search bytes and dev-selected weights before prospective reads."""
import argparse,json,subprocess
from pathlib import Path
import ram04_core as c
FILES=('config.json','ProbeServer.lean','ProbeServer.olean','ram04_core.py','model.py','train_local.py','runtime.py','search.py','evaluate.py')
def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();root=Path(__file__).parent;t=json.loads((a.training/'training_report.json').read_text())
    assert t['status']=='complete'
    selected={k:dict(seed=v['seed'],sha256=c.digest(v['checkpoint'])) for k,v in t['selected'].items()}
    assert all(selected[k]['sha256']==v['checkpoint_sha256'] for k,v in t['selected'].items())
    receipt=dict(implementation_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
      files={name:c.digest(root/name) for name in FILES},selected_models=selected,dataset_sha256=t['data']['dataset_sha256'],
      prospective_goal_reads_authorized_only_after_this_freeze=True)
    c.write_json(a.out,receipt);print('RAM04_PRE_EVALUATION_FREEZE',receipt['implementation_commit'],flush=True)
if __name__=='__main__':main()
