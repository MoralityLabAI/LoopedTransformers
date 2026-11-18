"""Offline tokenizer-only audit after inference, never loading model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from .prompt import tokens,aliases

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--compactification-root',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(a.compactification_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    tok=load_tokenizer(Path(a.snapshot),166100);runs=Path(a.runs)
    public={r['case_id']:r for r in map(json.loads,(runs/'cohort/public.jsonl').read_text().splitlines())}
    assert aliases(tok)==json.loads((runs/'cohort/source_receipt.json').read_text())['aliases']
    checked=steps=0
    for backend in ('bf16','nf4'):
        for cid,row in public.items():
            tape=json.loads((runs/backend/(cid.replace(':','_')+'.json')).read_text())
            for arm,r in tape['records'].items():
                variant='erased' if arm=='erased_repeat' else arm
                assert r['initial_prompt_ids']==tokens(tok,a.snapshot,row,variant)
                for i,s in enumerate(r['steps']):
                    assert s['decoded_prefix']==tok.tok.decode(r['generated_ids'][:i+1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    steps+=1
                checked+=1
    result={'status':'passed','model_weights_loaded':False,'case_arm_records_verified':checked,'decoded_prefixes_verified':steps,
        'checkpoint_revision':'451ed48c3273ecef7ea8faaa43c31ce529763bb1','sealed_test_accessed':False,
        'tokenizer_config_sha256':hashlib.sha256((Path(a.snapshot)/'tokenizer_config.json').read_bytes()).hexdigest()}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))

if __name__=='__main__':main()
