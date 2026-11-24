"""Offline prompt/gold trace lengths and ID bindings; no checkpoint tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from .common import ROOT,read,rows,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);a=p.parse_args()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['HF_HUB_OFFLINE']='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    from transformers import PreTrainedTokenizerFast
    from experiments.nan_gym05.common import prompt_ids
    snapshot=Path(a.snapshot);tok=SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(snapshot/'tokenizer.json')))
    cfg=read(ROOT/'config.json');public=rows(ROOT/'fixtures/public.jsonl');gold={r['id']:r for r in rows(ROOT/'fixtures/host_labels.jsonl')}
    records={}
    for row in public:
        ids=prompt_ids(tok,snapshot,row['prompt']);reply=tok.tok.encode(gold[row['id']]['canonical_trace'],add_special_tokens=False)
        assert len(ids)<=cfg['maximum_prompt_tokens'] and len(reply)<=cfg['maximum_generated_tokens']
        records[row['id']]=dict(tokens=len(ids),ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),canonical_trace_tokens=len(reply))
    write(ROOT/'fixtures/tokenizer_receipt.json',dict(contexts=records,
        tokenizer_files_sha256={name:sha(snapshot/name) for name in ('tokenizer.json','tokenizer_config.json')},
        public_sha256=sha(ROOT/'fixtures/public.jsonl'),maximum_prompt_tokens=max(v['tokens'] for v in records.values()),
        maximum_canonical_trace_tokens=max(v['canonical_trace_tokens'] for v in records.values()),pretrained_weights_loaded=False,sealed_test_accessed=False))
    print(dict(contexts=len(records),maximum_prompt_tokens=max(v['tokens'] for v in records.values()),
        maximum_canonical_trace_tokens=max(v['canonical_trace_tokens'] for v in records.values()),pretrained_model_calls=0))


if __name__=='__main__':
    main()
