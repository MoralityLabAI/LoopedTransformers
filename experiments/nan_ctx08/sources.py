"""Exact fresh ledgers, two source worlds, lookup/join and literal-copy controls."""
import argparse
from itertools import product
from pathlib import Path
import random
from types import SimpleNamespace
from experiments.nan_fit07.io import read,sha,write,write_rows
from experiments.nan_gym05.common import prompt_ids


LETTERS='ABCDEFGHIJKLMNOP'
HEADER=('Read the ledger and answer the question using only its records. Each key has one next key and one label. '
    'Reply with only the final label A through P.\n'
    'Example: X has next Y and label H; Y has next Z and label D. '
    'label(X) is H; label(next(X)) is D.\n\nLedger:\n')


def build(seed,count,position,task,world,copy_control):
    rng=random.Random(seed+count*1009);keys=['R'+str(i) for i in rng.sample(range(100000,1000000),count)]
    cycle=list(keys);rng.shuffle(cycle);nxt={k:cycle[(i+1)%count] for i,k in enumerate(cycle)}
    labels={k:rng.choice(LETTERS) for k in keys};query=cycle[0];dest=nxt[query]
    target=query if task=='lookup' else dest
    if world=='b':labels[target]=LETTERS[(LETTERS.index(labels[target])+8)%16]
    answer=labels[target];index=int(position*(count-1));other=count-1-index
    if other==index:other=(index+max(1,count//3))%count
    order=[k for k in keys if k not in (query,dest)];rng.shuffle(order)
    arranged=[None]*count;arranged[index]=query;arranged[other]=dest;remaining=iter(order)
    arranged=[next(remaining) if k is None else k for k in arranged]
    text=HEADER+'\n'.join(f'Record {k}: next={nxt[k]}; label={labels[k]}.' for k in arranged)
    question=f'label({query})' if task=='lookup' else f'label(next({query}))'
    text+='\n\nQuestion: '+question+'\n'
    if copy_control:text+='Copy control: ignore the question and copy exactly this supplied label: '+answer+'\n'
    text+='Answer:'
    return text,{'answer':answer,'query_key':query,'changed_key':target,'record_count':count,
        'query_record_index':index,'next_record_index':other,'world':world,'task':task,'copy_control':copy_control}


def variants(cfg):return list(product(cfg['positions'],cfg['tasks'],cfg['worlds'],(False,True)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);a=p.parse_args()
    root=Path(__file__).parent;cfg=read(root/'config.json');folder=root/'fixtures'
    if folder.exists():raise FileExistsError('Preserve generated source fixtures')
    import os
    cache=Path('runs/nan_ctx08/tokenizer_cache');cache.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    from transformers import PreTrainedTokenizerFast
    tok=SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(a.snapshot)/'tokenizer.json')))
    public=[];gold=[];tokens={};all_variants=variants(cfg)
    seeds=[('calibration',cfg['calibration_seed'])]+[('development',cfg['development_first_seed']+i) for i in range(cfg['development_roots'])]
    for split,seed in seeds:
        for budget in cfg['context_budgets']:
            selected=all_variants if split=='development' else [('middle','lookup','a',False)]
            # Choose count without any model inference or answer-based filtering.
            # One count per source root/budget fits every world/task/position/control.
            def longest(count):
                return max(len(prompt_ids(tok,a.snapshot,build(seed,count,cfg['positions'][pos],task,w,c)[0])) for pos,task,w,c in all_variants)
            lo,hi=8,512
            while lo<hi:
                mid=(lo+hi+1)//2
                if longest(mid)<=budget:lo=mid
                else:hi=mid-1
            for pos,task,world,control in selected:
                text,label=build(seed,lo,cfg['positions'][pos],task,world,control)
                ids=prompt_ids(tok,a.snapshot,text);length=len(ids)
                if not budget-cfg['maximum_underfill_tokens']<=length<=budget:raise ValueError('Registered context band not met')
                key=f'ctx08:{seed}:{budget}:{pos}:{task}:{world}:'+('copy' if control else 'main')
                public.append(dict(id=key,root=f'ctx08:{seed}',split=split,context_budget=budget,position=pos,task=task,world=world,
                    copy_control=control,prompt=text,schema=dict(type='label',allowed=list(LETTERS),max_characters=64)))
                gold.append(dict(id=key,**label))
                import hashlib,json
                tokens[key]={'tokens':length,'initial_prompt_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    write_rows(folder/'public.jsonl',public);write_rows(folder/'host_labels.jsonl',gold)
    write(folder/'tokenizer_receipt.json',{'pretrained_weights_loaded':False,'public_sha256':sha(folder/'public.jsonl'),
        'tokenizer_config_sha256':sha(Path(a.snapshot)/'tokenizer_config.json'),'contexts':tokens,
        'source_only_count_selection':'maximum over all registered variants; no model observations',
        'sealed_test_accessed_or_used_as_exclusion':False})
    print({'contexts':len(public),'DEV_contexts':sum(r['split']=='development' for r in public),
        'context_budgets':cfg['context_budgets'],'minimum_tokens':min(r['tokens'] for r in tokens.values()),'maximum_tokens':max(r['tokens'] for r in tokens.values())})


if __name__=='__main__':main()
