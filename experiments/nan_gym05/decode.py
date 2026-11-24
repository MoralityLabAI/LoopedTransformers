"""Native full-prefix greedy generation; first-prefix read-only tower recording."""
import hashlib
import json
import torch
from .parser import parse

def digest(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

@torch.inference_mode()
def generate(observer,head,tok,tasks,sequences,cfg):
    records={r['id']:{'schema':r['schema'],'initial_prompt_ids':sequences[r['id']],
        'initial_prompt_sha256':digest(sequences[r['id']]),'generated_ids':[],'steps':[]} for r in tasks}
    caps={r['id']:cfg['maximum_'+r['schema']['type']+'_tokens'] for r in tasks}
    active=list(records);observations={}
    for index in range(max(caps.values())):
        if not active:break
        prefixes={n:sequences[n]+records[n]['generated_ids'] for n in active}
        states,values,one=observer.forward(prefixes,capture=index==0)
        if index==0:observations=values
        for j,(name,ids,h) in enumerate(zip(active,prefixes.values(),states)):
            r=records[name];score=head.score(h);token=score['free_top_token_id'];r['generated_ids'].append(token)
            if index==0:r['first_prefix_single_pass_score']=head.score(one[j])
            raw=tok.decode(r['generated_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            ended=token==tok.eos_token_id
            text=tok.decode(r['generated_ids'][:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False) if ended else raw
            r['parser']=parse(text,r['schema'],final=ended or index+1==caps[name])
            r['steps'].append({'step':index+1,'prefix_sha256':digest(ids),'raw_decoded_prefix':raw,'parse_text':text,'score':score})
        active=[n for n in active if records[n]['parser']['status']=='pending']
        del states,one
    for r in records.values():
        r['generated_text']=r['steps'][-1]['parse_text'];r['tokens_generated']=len(r['generated_ids'])
        if r['parser']['status']=='pending':raise RuntimeError('Budget did not close parser')
    return records,observations
