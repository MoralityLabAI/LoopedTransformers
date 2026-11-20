"""Offline prompt and complete prefix detokenization audit; no model weights."""
import argparse
import json
from pathlib import Path
import sys
from .prompt import tokens

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(a.compactification_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    tok=load_tokenizer(Path(a.snapshot),166100);runs=Path(a.runs);rows=[json.loads(s) for s in (runs/'cohort/public.jsonl').read_text().splitlines()];count=steps=0
    for b in ('bf16','nf4'):
        for row in rows:
            tape=json.loads((runs/b/f'root_{row["root"]}.json').read_text())
            for arm,r in tape['records'].items():
                assert r['initial_prompt_ids']==tokens(tok,a.snapshot,row['prompts'][arm])
                for i,s in enumerate(r['steps']):
                    ids=r['generated_ids'][:i+1];assert s['raw_decoded_prefix']==tok.tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    parsed=tok.tok.decode(ids[:-1] if ids[-1]==tok.tok.eos_token_id else ids,skip_special_tokens=False,clean_up_tokenization_spaces=False);assert parsed==s['parse_text'];steps+=1
                count+=1
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);result={'status':'passed','model_weights_loaded':False,'prompt_bindings':count,'decoded_prefixes':steps,'sealed_test_accessed':False}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))

if __name__=='__main__':main()
