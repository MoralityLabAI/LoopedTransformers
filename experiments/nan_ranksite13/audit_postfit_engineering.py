"""Independent all-cell CPU validation of the random full-tape post-fit path."""
import argparse
from pathlib import Path
import numpy as np
from .common import read,rows,sha,write,condition,key
from .physical_audit import physical_audit
from .resources import below_normal


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args();below_normal()
    runs=Path(a.runs);root=Path(__file__).parent;proof=read(runs/'receipt.json');assert proof['status']=='passed' and not proof['pretrained_weights_loaded']
    for name,h in proof['source_sha256'].items():assert sha(root/name)==h,name
    assert proof['native_bridge_sha256']==sha('mechinterp/nanbeige_bpd_gym/o_proj_bridge.py')
    public=rows('experiments/nan_fit07/fixtures/public.jsonl');selected=[next(r for r in public if r['split']==s and not r['copy_control']) for s in ('calibration','development')]
    rng=np.random.default_rng(2026112013)
    sae=dict(encoder=rng.normal(size=(128,64))/8,bias=np.zeros(128),decoder=rng.normal(size=(64,128))/np.sqrt(128),
        decoder_bias=np.zeros(64),mean=np.zeros(64),scale=np.ones(64))
    result=[]
    for index in range(6):
        cfg=condition(index);cfg.update(input_width=128,output_width=64,physical_probe_main_contexts=2)
        folder=runs/'conditions'/key(cfg);fit=read(folder/'receipt.json')
        assert fit['target_parameter_sha256_unchanged']==proof['target_parameter_sha256']
        assert sha(folder/'physical_bridge/receipt.json')==proof['results'][index]['physical_receipt_sha256']
        checked=physical_audit(folder,selected,cfg,fit,sae)
        assert checked['status']=='passed' and len(checked['geometry_groups'])==72 and len(checked['derivative_diagnostics'])==36
        for group in checked['geometry_groups']:
            for field in ('encoder_cosine','decoder_cosine','encoder_preactivation_delta','SAE_code_delta'):
                assert len(group['root_macro_over_all_components'][field])==128
        result.append(dict(condition=key(cfg),status='passed',signed_finite_writes_verified=checked['signed_finite_writes_verified'],
            physical_receipt_sha256=checked['physical_receipt_sha256'],geometry_groups=len(checked['geometry_groups']),
            derivative_groups=len(checked['derivative_diagnostics']),features_retained=128))
        print(result[-1],flush=True)
    payload=dict(status='passed',cells=result,signed_finite_writes_verified=sum(r['signed_finite_writes_verified'] for r in result),
        source_sha256=sha(Path(__file__)),physical_auditor_sha256=sha(root/'physical_audit.py'),engineering_receipt_sha256=sha(runs/'receipt.json'),
        pretrained_weights_loaded=False,SAE_scope='random width64/128-feature engineering dictionary; not a fitted or pretrained feature claim',
        sealed_test_accessed=False,BPD_donor_transfer_attempted=False,formal_inferential_comparisons=0)
    write(a.out,payload);print({'status':'passed','signed_finite_writes_verified':payload['signed_finite_writes_verified']})


if __name__=='__main__':main()
