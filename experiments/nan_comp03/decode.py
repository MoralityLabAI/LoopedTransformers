"""B1 grouped layer streaming; independent greedy generation without KV caching."""
import hashlib
import json
import torch
from .parser import parse

def token_hash(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

@torch.inference_mode()
def grouped_states(loop,sequences):
    prepared=[loop.prepare(torch.tensor([ids],dtype=torch.long)) for ids in sequences]
    one=loop.run_pairs(prepared)
    two=loop.run_pairs([(h,ctx) for h,(_,ctx) in zip(one,prepared)])
    return two

@torch.inference_mode()
def generate(loop,head,tok,sequences,*,max_tokens=8):
    records={name:{'initial_prompt_ids':list(ids),'initial_prompt_sha256':token_hash(ids),'generated_ids':[],'steps':[],
                   'parser':{'status':'pending','operator':None,'reason':'grammar_prefix'}} for name,ids in sequences.items()}
    active=list(sequences)
    for step in range(max_tokens):
        if not active:break
        prefixes=[sequences[name]+records[name]['generated_ids'] for name in active]
        states=grouped_states(loop,prefixes)
        for name,ids,h in zip(active,prefixes,states):
            r=records[name];score=head.score(h);token=score['free_top_token_id'];r['generated_ids'].append(token)
            text=tok.decode(r['generated_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            ended=token==tok.eos_token_id
            parsed=parse(text,final=ended or step+1==max_tokens)
            r['steps'].append({'step':step+1,'prefix_sha256':token_hash(ids),'decoded_prefix':text,'score':score})
            r['parser']=parsed
            if parsed['status']!='pending':r['stop_reason']='eos' if ended else ('token_limit' if step+1==max_tokens and parsed['reason']=='incomplete_at_limit' else parsed['reason'])
        active=[name for name in active if records[name]['parser']['status']=='pending']
        del states
    for r in records.values():
        r['generated_text']=r['steps'][-1]['decoded_prefix'];r['tokens_generated']=len(r['generated_ids'])
    return records
