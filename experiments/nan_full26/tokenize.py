"""Tokenization only, with all model backends disabled before import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from .common import ROOT,THREADS,load_sources,sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);a=p.parse_args()
    for key in THREADS:os.environ[key]='1'
    os.environ.update(USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    from transformers import PreTrainedTokenizerFast
    from experiments.nan_gym05.common import prompt_ids
    snapshot=Path(a.snapshot);tok=SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(snapshot/'tokenizer.json')))
    public,pins=load_sources();records={}
    for r in public:
        ids=prompt_ids(tok,snapshot,r['prompt']);assert len(ids)+256<=9216
        records[r['id']]=dict(tokens=len(ids),ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    write(ROOT/'tokenizer_receipt.json',dict(contexts=records,source_pins=pins,
        tokenizer_files_sha256={n:sha(snapshot/n) for n in ('tokenizer.json','tokenizer_config.json')},
        maximum_prompt_tokens=max(v['tokens'] for v in records.values()),pretrained_weights_loaded=False,sealed_test_accessed=False))
    print(dict(contexts=len(records),maximum_prompt_tokens=max(v['tokens'] for v in records.values()),pretrained_model_calls=0))

if __name__=='__main__':main()
