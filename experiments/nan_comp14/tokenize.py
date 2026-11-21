"""Tokenize source prompts offline before freezing; load no model tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from .common import ROOT, read, rows, sha, write_new


def main():
    p = argparse.ArgumentParser(); p.add_argument('--snapshot', required=True)
    p.add_argument('--compactification-root', required=True); a = p.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ.pop('TRANSFORMERS_CACHE', None)
    sys.path.insert(0, str(Path(a.compactification_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    from experiments.nan_gym05.common import prompt_ids
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); tokenizer = load_tokenizer(Path(a.snapshot), 166100)
    cfg = read(ROOT / 'config.json'); records = {}
    for row in rows(ROOT / 'fixtures/public.jsonl'):
        ids = prompt_ids(tokenizer, a.snapshot, row['prompt'])
        assert len(ids) <= cfg['maximum_prompt_tokens']
        records[row['id']] = dict(tokens=len(ids), ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    write_new(ROOT / 'fixtures/tokenizer_receipt.json', dict(contexts=records,
        tokenizer_config_sha256=sha(Path(a.snapshot) / 'tokenizer_config.json'),
        maximum_prompt_tokens=max(v['tokens'] for v in records.values()),
        pretrained_model_loaded=False, sealed_test_accessed=False))
    print({'contexts': len(records), 'maximum_prompt_tokens': max(v['tokens'] for v in records.values())})


if __name__ == '__main__':
    main()
