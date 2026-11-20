"""Full-prefix native two-loop greedy generation, independent B1 contexts."""
import hashlib
import json
import torch
from experiments.nan_comp03.decode import grouped_states
from .parser import parse

def digest(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

@torch.inference_mode()
def generate(loop,head,tok,sequences,kinds,cfg):
    records={name:{'kind':kinds[name],'initial_prompt_ids':ids,'initial_prompt_sha256':digest(ids),'generated_ids':[],'steps':[],
        'parser':{'status':'pending','value':None,'reason':'grammar_prefix'}} for name,ids in sequences.items()};active=list(records)
    for index in range(cfg['maximum_integer_tokens']):
        if not active:break
        prefixes=[sequences[name]+records[name]['generated_ids'] for name in active];states=grouped_states(loop,prefixes)
        for name,ids,h in zip(active,prefixes,states):
            r=records[name];score=head.score(h);token=score['free_top_token_id'];r['generated_ids'].append(token)
            raw=tok.decode(r['generated_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            ended=token==tok.eos_token_id
            text=tok.decode(r['generated_ids'][:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False) if ended else raw
            cap=cfg['maximum_integer_tokens'] if r['kind']=='integer' else cfg['maximum_operator_tokens']
            r['parser']=parse(text,r['kind'],eos=ended,limit=index+1==cap)
            r['steps'].append({'step':index+1,'prefix_sha256':digest(ids),'raw_decoded_prefix':raw,'parse_text':text,'score':score})
            if r['parser']['status']!='pending':r['stop_reason']='eos' if ended else r['parser']['reason']
        active=[name for name in active if records[name]['parser']['status']=='pending'];del states
    for r in records.values():
        r['tokens_generated']=len(r['generated_ids']);r['generated_text']=r['steps'][-1]['parse_text']
    return records
