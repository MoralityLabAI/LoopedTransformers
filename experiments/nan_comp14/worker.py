"""Registered full-prefix native inference; write-once cases, no host gold."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from .common import ROOT, RUNS, read, rows, sha, verify, write_new
from .parser import label


def main():
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--compactification-root', required=True)
    p.add_argument('--runs', default=str(RUNS)); a = p.parse_args()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ.pop('TRANSFORMERS_CACHE', None)
    out = Path(a.runs) / a.backend; out.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HOME'] = str((out / 'HF_cache').resolve())
    import numpy as np
    import torch
    from experiments.nan_bpd01.backend import load_backend
    from experiments.nan_comp02.head import OperatorHead
    from experiments.nan_comp03.prompt import aliases
    from experiments.nan_gym05.common import prompt_ids
    from experiments.nan_gym05.observe import TowerObserver
    from experiments.nan_ranksite13.resources import below_normal, driver_sample
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
    from mechinterp.nanbeige_bpd_gym.resident_head import mount, retain_decoder_layers, check_parity
    below_normal(); seal = verify(); cfg = read(ROOT / 'config.json')
    assert subprocess.check_output(['git', '-C', str(Path(a.compactification_root).resolve()), 'rev-parse', 'HEAD'], text=True).strip() == seal['compactification_adapter_commit']
    source = rows(ROOT / 'fixtures/public.jsonl'); tokens = read(ROOT / 'fixtures/tokenizer_receipt.json')
    assert len(source) == 240 and sha(Path(a.snapshot) / 'tokenizer_config.json') == tokens['tokenizer_config_sha256']
    torch.set_num_threads(1); torch.manual_seed(cfg['source_seed'])
    torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); completed = 0; failure = None; status = 'failed'; observer = None

    def event(name, **fields):
        v = dict(event=name, **fields)
        with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(v) + '\n')
        print(v, flush=True)

    def budget():
        if time.monotonic() - started > cfg['maximum_backend_seconds']:
            raise TimeoutError('Registered between-prefix backend budget reached; no retries')

    try:
        assert_oracle_isolated()
        loop, legacy, tok, _ = load_backend(a.compactification_root, a.snapshot, a.backend, chunk_rows=cfg['scorer_chunk_rows'])
        if loop.stage_layer is not None:
            loop.stage_layer.requires_grad_(False).eval()
        if a.backend == 'bf16':
            retain_decoder_layers(loop, cfg['BF16_resident_decoder_layers'])
        cpu = OperatorHead(legacy.weight, aliases(tok), chunk_rows=cfg['scorer_chunk_rows'])
        head = mount(loop, cpu); del legacy
        before = target_digest(loop); observer = TowerObserver(loop, cfg['layers_zero_based'])
        write_new(out / 'manifest.json', dict(backend=a.backend, pid=os.getpid(),
            source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), checkpoint_revision=cfg['checkpoint_revision'],
            CPU_threads=torch.get_num_threads(), priority='BelowNormal', decoder_GPU_layers=loop.n_gpu_layers,
            head_device=str(head.weight.device), head_dtype=str(head.weight.dtype),
            embedding_device=str(loop.base.embed_tokens.weight.device), target_parameter_sha256=before,
            host_gold_opened=False, sealed_test_accessed=False))
        write_new(out / 'driver_after_load.json', driver_sample())
        for ordinal, row in enumerate(source):
            budget(); ids = prompt_ids(tok, a.snapshot, row['prompt']); bound = tokens['contexts'][row['id']]
            assert len(ids) == bound['tokens'] and hashlib.sha256(json.dumps(ids).encode()).hexdigest() == bound['ids_sha256']
            begin = time.monotonic(); generated = []; steps = []; initial = []; state_index = []; text = ''; parsed = None
            for step in range(cfg['maximum_generated_tokens']):
                budget()
                with torch.inference_mode():
                    two, states, one = observer.forward({'context': ids + generated}, capture=step == 0)
                    if step == 0:
                        if cpu is not None:
                            parity = [check_parity(cpu, head, h) for h in (one[0], two[0])]
                            write_new(out / 'head_parity.json', dict(checks=parity)); del cpu; cpu = None; gc.collect()
                        arrays = {}
                        for j, ((_, phase, layer, site), value) in enumerate(states.items()):
                            key = 'v' + str(j); arrays[key] = value.numpy()
                            state_index.append(dict(loop=phase, layer_zero_based=layer, site=site, array_key=key))
                        for phase, h in enumerate((one[0], two[0]), 1):
                            scored = head.score(h); token = scored['free_top_token_id']
                            initial.append(dict(loop=phase, score=scored, token_text=tok.tok.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)))
                        statefile = out / f'case_{ordinal:04d}_states.npz'
                        with statefile.open('xb') as f:
                            np.savez_compressed(f, **arrays)
                        assert len(state_index) == 40
                        del arrays
                    else:
                        scored = head.score(two[0])
                    if step == 0:
                        scored = initial[1]['score']
                    token = scored['free_top_token_id']; generated.append(token)
                    text = tok.tok.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    steps.append(dict(step=step, history_tokens=len(ids) + step, generated_token_id=token,
                                      decoded_prefix=text, score=scored))
                    parsed = label(text)
                    del two, one, states
                if parsed is not None or token == tok.tok.eos_token_id or len(text) > cfg['maximum_reply_characters']:
                    break
            record = {k: row[k] for k in ('id', 'root', 'split', 'world', 'arm')}
            record.update(initial_prompt_ids=ids, initial_prompt_ids_sha256=bound['ids_sha256'], initial_phases=initial,
                generated_ids=generated, steps=steps, bounded_reply=text, parsed_label=parsed,
                state_file=statefile.name, state_file_sha256=sha(statefile), state_index=state_index,
                elapsed_seconds=time.monotonic() - begin, CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(), status='completed')
            write_new(out / f'case_{ordinal:04d}.json', record)
            completed += 1; event('case_closed', ordinal=ordinal, completed_contexts=completed)
        assert target_digest(loop) == before; assert_oracle_isolated(); status = 'completed'
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error)); traceback.print_exc()
    finally:
        if observer is not None:
            observer.close()
        write_new(out / 'driver_terminal.json', driver_sample())
        write_new(out / 'receipt.json', dict(backend=a.backend, status=status, failure=failure,
            source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), completed_contexts=completed,
            registered_contexts=len(source), elapsed_seconds=time.monotonic() - started,
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            host_gold_opened=False, sealed_test_accessed=False, BPD_donor_transfer_attempted=False))
        event('worker_receipt_closed', status=status)
    if status != 'completed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
