"""Registered native no-cache prefixes; host oracle never imported or opened."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from .common import ROOT, RUNS, read, rows, sha, verify, write

FIELDS = ('id', 'root', 'split', 'context_budget', 'position', 'world', 'arm', 'graph_records', 'target_graph_index')


class BudgetReached(RuntimeError):
    pass


def main():
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--compactification-root', required=True); a = p.parse_args()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ.pop('TRANSFORMERS_CACHE', None)
    from experiments.nan_ranksite13.resources import below_normal, driver_sample
    below_normal(); seal = verify(); cfg = read(ROOT/'config.json'); started = time.monotonic()
    assert subprocess.check_output(['git', '-C', a.compactification_root, 'rev-parse', 'HEAD'], text=True).strip() == seal['adapter_commit']
    out = RUNS/a.backend; out.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HOME'] = str((out/'HF_cache').resolve())
    import numpy as np
    import torch
    from experiments.nan_bpd01.backend import load_backend
    from experiments.nan_comp02.head import OperatorHead
    from experiments.nan_comp03.prompt import aliases
    from experiments.nan_gym05.common import prompt_ids
    from experiments.nan_gym05.observe import TowerObserver
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
    from mechinterp.nanbeige_bpd_gym.resident_head import mount, retain_decoder_layers, check_parity
    public = rows(ROOT/'fixtures/public.jsonl'); token_receipt = read(ROOT/'fixtures/tokenizer_receipt.json')
    assert len(public) == 624
    for name, expected in token_receipt['tokenizer_files_sha256'].items():
        assert sha(Path(a.snapshot)/name) == expected
    torch.set_num_threads(1); torch.manual_seed(cfg['source_seed'])
    status = 'failed'; failure = None; completed = 0; observer = None; before = after = None; initialized = False

    def event(name, **fields):
        value = dict(event=name, **fields)
        with (out/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value)+'\n')
        print(value, flush=True)

    def budget():
        if time.monotonic() - started > cfg['maximum_backend_seconds']:
            raise BudgetReached('Four-hour registered between-prefix budget reached; no retry')

    try:
        torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']); torch.cuda.reset_peak_memory_stats(); initialized = True
        assert_oracle_isolated(); budget()
        loop, legacy, tok, _ = load_backend(a.compactification_root, a.snapshot, a.backend, chunk_rows=cfg['scorer_chunk_rows'])
        if loop.stage_layer is not None:
            loop.stage_layer.requires_grad_(False).eval()
        if a.backend == 'bf16':
            retain_decoder_layers(loop, cfg['BF16_resident_decoder_layers'])
        cpu = OperatorHead(legacy.weight, aliases(tok), chunk_rows=cfg['scorer_chunk_rows'])
        head = mount(loop, cpu); del legacy
        before = target_digest(loop); observer = TowerObserver(loop, cfg['layers_zero_based'])
        write(out/'manifest.json', dict(backend=a.backend, pid=os.getpid(), source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
            checkpoint_revision=cfg['checkpoint_revision'], target_parameter_sha256=before, CPU_threads=torch.get_num_threads(),
            priority='BelowNormal', decoder_GPU_layers=loop.n_gpu_layers, embedding_device=str(loop.base.embed_tokens.weight.device),
            head_device=str(head.weight.device), head_dtype=str(head.weight.dtype), host_gold_opened=False, sealed_test_accessed=False))
        write(out/'driver_after_load.json', driver_sample()); event('model_loaded', backend=a.backend)
        for ordinal, row in enumerate(public):
            budget(); ids = prompt_ids(tok, a.snapshot, row['prompt']); bound = token_receipt['contexts'][row['id']]
            assert len(ids) == bound['tokens'] and hashlib.sha256(json.dumps(ids).encode()).hexdigest() == bound['ids_sha256']
            assert row['context_budget'] - cfg['maximum_underfill_tokens'] <= len(ids) <= row['context_budget']
            begin = time.monotonic()
            with torch.inference_mode():
                two, states, one = observer.forward({'context':ids}, capture=True)
                if cpu is not None:
                    parity = [check_parity(cpu, head, h) for h in (one[0], two[0])]
                    write(out/'head_parity.json', dict(checks=parity)); del cpu; cpu = None; gc.collect()
                phases = []
                for phase, h in enumerate((one[0], two[0]), 1):
                    score = head.score(h)
                    phases.append(dict(loop=phase, score=score, token_text=tok.tok.decode([score['free_top_token_id']], skip_special_tokens=False, clean_up_tokenization_spaces=False)))
                arrays = {}; index = []
                for j, ((_, phase, layer, site), value) in enumerate(states.items()):
                    key = 'v'+str(j); arrays[key] = value.numpy()
                    index.append(dict(loop=phase, layer_zero_based=layer, site=site, array_key=key))
                assert len(index) == 40
                statefile = out/f'case_{ordinal:04d}_states.npz'
                with statefile.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                record = {k:row[k] for k in FIELDS}
                record.update(phases=phases, initial_prompt_ids=ids, initial_prompt_ids_sha256=bound['ids_sha256'],
                    state_file=statefile.name, state_file_sha256=sha(statefile), state_index=index,
                    elapsed_seconds=time.monotonic()-begin, CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(), status='completed')
                write(out/f'case_{ordinal:04d}.json', record)
                del two, one, h, states, arrays, value
            completed += 1; event('case_closed', ordinal=ordinal, completed_contexts=completed, context_budget=row['context_budget'])
        after = target_digest(loop); assert after == before; assert_oracle_isolated(); status = 'completed'
    except torch.cuda.OutOfMemoryError as error:
        status = 'capacity_stopped'; failure = dict(type=type(error).__name__, message=str(error)); traceback.print_exc()
    except Exception as error:
        status = 'budget_reached' if isinstance(error, BudgetReached) else 'failed'
        failure = dict(type=type(error).__name__, message=str(error)); traceback.print_exc()
    finally:
        if observer is not None:
            observer.close()
        write(out/'driver_terminal.json', driver_sample())
        levels = [dict(context_budget=n, registered_contexts=sum(r['context_budget']==n for r in public),
            completed_contexts=sum(r['context_budget']==n for r in public[:completed])) for n in cfg['context_budgets']]
        write(out/'receipt.json', dict(backend=a.backend, status=status, failure=failure, completed_contexts=completed,
            registered_contexts=624, levels=levels, source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
            elapsed_seconds=time.monotonic()-started, failed_or_next_ordinal=completed if completed<624 else None,
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated() if initialized else None,
            CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved() if initialized else None,
            target_parameter_sha256_before=before, target_parameter_sha256_after=after,
            host_gold_opened=False, BPD_donor_transfer_attempted=False, sealed_test_accessed=False))
        event('worker_receipt_closed', status=status)
    if status == 'failed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
