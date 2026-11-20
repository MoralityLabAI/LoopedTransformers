"""Host-only clean-competence gate. Never imported by the model worker."""
from pathlib import Path
from experiments.nan_gym05.parser import parse
from .io import read,rows,sha,write


def grade(baseline,public,labels,cfg):
    by={r['id']:r for r in public};gold={r['id']:r for r in labels};seen=set();groups={}
    for row in baseline['records']:
        key=row['id'];assert key not in seen;seen.add(key);source=by[key]
        label=gold[key];condition='copy' if source['copy_control'] else 'main'
        value=parse(row['phases'][1]['free_token_text'],source['schema'],final=True)['value']
        group=groups.setdefault((source['split'],condition),{'records':0,'correct':0})
        group['records']+=1;group['correct']+=value==label['answer']
    assert seen==set(by)==set(gold) and len(seen)==32
    passed=all(groups[s,'main']['records']==groups[s,'copy']['records']==8 and
        groups[s,'main']['correct']>=cfg['minimum_clean_correct_per_split'] and
        groups[s,'copy']['correct']==cfg['required_clean_copy_correct_per_split'] for s in ('calibration','development'))
    return {'allow_fits':passed,'scope':'first free token of native two-loop prefix; no case filtering; no causal-transfer permission',
        'groups':[{'split':s,'condition':c,**v} for (s,c),v in sorted(groups.items())],
        'original_algebraic_gate_unchanged':True,'BPD_donor_transfer_enabled':False,'sealed_test_accessed':False}


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=('nf4','bf16'));p.add_argument('--runs',default='runs/nan_fit07');a=p.parse_args()
    root=Path(__file__).parent;folder=Path(a.runs)/a.backend;baseline=folder/'baseline.json'
    seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['host_only_files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Frozen host-side source changed: '+name)
    if read(baseline)['source_manifest_sha256']!=sha(root/'SOURCE_MANIFEST.json'):raise ValueError('Baseline registration differs')
    result=grade(read(baseline),rows(root/'fixtures/public.jsonl'),rows(root/'fixtures/host_labels.jsonl'),read(root/'config.json'))
    result|={'baseline_sha256':sha(baseline),'public_sha256':sha(root/'fixtures/public.jsonl'),'host_labels_sha256':sha(root/'fixtures/host_labels.jsonl')}
    write(folder/'host_gate.json',result);print(result)


if __name__=='__main__':main()
