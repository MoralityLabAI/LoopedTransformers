"""Audit actual official Prime traces against fresh local worker evidence.

Run in the pinned model environment for tokenizer-only decoding (no model load,
no CUDA operations). All per-record material stays local; output is aggregate.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import numpy as np
from experiments.nan_gym05.common import read,rows,sha,write
from experiments.nan_prime06.wire import prompt_sha
from .suite_rows import load_suite_rows
from .audit_native import final_answer,correctness


def sudoku_valid(value,givens):
    if not isinstance(value,list) or len(value)!=4 or any(type(r) is not list or len(r)!=4 for r in value):return 0
    if any(type(x) is not int for r in value for x in r):return 0
    target={1,2,3,4}
    if any(set(r)!=target for r in value) or any({value[i][j] for i in range(4)}!=target for j in range(4)):return 0
    if any({value[i][j] for i in range(a,a+2) for j in range(b,b+2)}!=target for a in (0,2) for b in (0,2)):return 0
    return int(all(not givens[i][j] or givens[i][j]==value[i][j] for i in range(4) for j in range(4)))


def audit(runs,backend,snapshot):
    from transformers import AutoTokenizer
    root=Path('experiments/nan_prime06');cfg=read(root/'config.json');seal=read(root/'SOURCE_MANIFEST.json')
    assert sha(root/'SOURCE_MANIFEST.json')=='98973e25cfa188d0a7f67bd2ebcc30433efd8145e9bb8bd358ef52aa589a7b47'
    for name,h in seal['files_sha256'].items():assert sha(root/name)==h
    for name,h in seal['dependencies_sha256'].items():assert sha(name)==h
    folder=runs/backend;receipt=read(folder/'receipt.json');manifest=read(folder/'manifest.json')
    assert receipt['status']=='completed' and receipt['requests']==162 and not manifest['mock_only']
    assert manifest['source_registration_sha256']==sha(root/'SOURCE_MANIFEST.json')
    expected=load_suite_rows(scope='local',include_copy_controls=True)
    episodes_file=runs/'prime'/('nan-prime06-'+backend)/'traces.jsonl';episodes=rows(episodes_file)
    files=sorted(folder.glob('request_*.json'));assert len(episodes)==len(files)==len(expected)==162
    source=read('experiments/nan_gym05/SOURCE_MANIFEST.json')
    assert sha(snapshot/'tokenizer_config.json')==source['tokenizer_config_sha256']
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),trust_remote_code=True,local_files_only=True)
    native={}
    for file in Path('runs/nan_gym05',backend).glob('root_*.json'):native.update(read(file)['records'])
    groups=defaultdict(list);state_vectors=0;tokens=0;prefix_agreements=0
    for i,(row,e,file) in enumerate(zip(expected,episodes,files)):
        assert e['ok'] and not e['is_timeout'] and not e['errors'] and len(e['traces'])==1
        t=e['traces'][0];assert t['ok'] and t['is_completed'] and not t['errors'] and not t['is_timeout']
        assert t['verifiers']['commit']==cfg['verifiers_commit'] and len(t['calls'])==t['num_turns']==1
        assert t['agent']['runtime']['type']=='subprocess' and t['agent']['config']['harness']['id']=='null'
        data=t['task']['data'];assert data['id']==row['id'] and data['prompt']==row['prompt'] and data['private_label']==row['private_label']
        assert len(t['nodes'])==2 and t['nodes'][0]['message']=={'role':'user','content':row['prompt']}
        record=read(file);assert not record['mock_only'] and record['completion_id']=='nan-prime06-'+backend+'-'+str(i)
        assert record['prompt_sha256']==prompt_sha(row['prompt']) and record['request']['messages']==[{'role':'user','content':row['prompt']}]
        assert t['last_reply']==record['text'] and t['nodes'][1]['message']['content']==record['text']
        assert record['initial_prompt_ids']==native[row['id']]['initial_prompt_ids']
        generated=record['generated_ids'];cap=cfg['maximum_'+row['schema']['type']+'_tokens']
        assert 1<=len(generated)<=cap and tokenizer.eos_token_id not in generated[:-1]
        ended=generated[-1]==tokenizer.eos_token_id
        assert ended or len(generated)==cap
        assert record['finish_reason']==('stop' if ended else 'length')==t['calls'][0]['finish_reason']
        decoded=tokenizer.decode(generated[:-1] if ended else generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        assert decoded==record['text'] and len(record['scores'])==len(generated)
        for token,score in zip(generated,record['scores']):
            assert token==score['free_top_token_id'] and 0<=token<166144 and math.isfinite(score['full_log_normalizer'])
        usage=record['usage'];assert usage['prompt_tokens']==len(record['initial_prompt_ids']) and usage['completion_tokens']==len(generated)
        assert t['num_input_tokens']==usage['prompt_tokens'] and t['num_output_tokens']==usage['completion_tokens']
        value=final_answer(record['text'],row['schema']);label=row['private_label'];score=correctness(value,label)
        assert score==t['reward'] and (value is not None)==t['info']['format_valid']
        secondary=sudoku_valid(value,label['givens']) if row['suite']=='sudoku' and not label['copy_control'] else None
        if secondary is not None:assert secondary==t['info']['valid_sudoku_solution']
        statefile=folder/record['state_file'];assert sha(statefile)==record['state_file_sha256']
        want={(phase,layer,site) for phase in (1,2) for layer in (5,11,17,21)
            for site in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
        actual={(o['loop'],o['layer_zero_based'],o['site']) for o in record['state_index']};assert actual==want and len(actual)==40
        with np.load(statefile,allow_pickle=False) as z:
            assert set(z.files)=={o['array_key'] for o in record['state_index']}
            for o in record['state_index']:
                v=z[o['array_key']];width={'down_proj_input':10752,'o_proj_input':6144}.get(o['site'],3072)
                assert v.shape==(width,) and v.dtype==np.float32 and np.isfinite(v).all();state_vectors+=1
        common=min(len(generated),len(native[row['id']]['generated_ids']))
        prefix_agreements+=int(generated[:common]==native[row['id']]['generated_ids'][:common])
        condition='copy' if label['copy_control'] else ('hop'+str(label['hops']) if label.get('hops') else 'main')
        groups[row['suite'],condition].append({'root':row['root_id'],'correct':score,'format':value is not None,'valid_sudoku':secondary,'ended':ended})
        tokens+=len(generated)
    summary=[]
    for key,rr in sorted(groups.items()):
        row={'suite':key[0],'condition':key[1],'contexts':len(rr),'source_roots':len({r['root'] for r in rr}),
            'correct':sum(r['correct'] for r in rr),'format_valid':sum(r['format'] for r in rr),'EOS_ended':sum(r['ended'] for r in rr)}
        if key==('sudoku','main'):row['valid_givens_preserving_solutions']=sum(r['valid_sudoku'] for r in rr)
        summary.append(row)
    return {'experiment':'NAN-PRIME-06','status':'passed','backend':backend,'episodes':162,'fresh_worker_requests':162,
        'state_vectors':state_vectors,'generated_tokens':tokens,'native_prefix_token_agreements':prefix_agreements,
        'scope':'actual local SDK full-reply rollouts; no pooling with prefix-stop accuracy',
        'formal_inferential_comparisons':0,'summaries':summary,'official_trace_file_sha256':sha(episodes_file),
        'worker_receipt_sha256':sha(folder/'receipt.json'),'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
        'BPD_transfer_attempted':False,'sealed_test_accessed':False,'per_record_routing_material_published':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_prime06');p.add_argument('--backend',choices=('bf16','nf4'),required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=audit(Path(a.runs),a.backend,Path(a.snapshot));write(a.out,result)
    print({k:result[k] for k in ('status','backend','episodes','fresh_worker_requests','state_vectors')})


if __name__=='__main__':main()
