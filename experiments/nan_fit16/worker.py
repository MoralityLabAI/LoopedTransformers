"""Actual stock optimizer fitting; no host oracle import or outcome selection."""
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


def main():
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--compactification-root', required=True)
    p.add_argument('--bpd-root', required=True); a = p.parse_args()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ.pop('TRANSFORMERS_CACHE', None)
    from experiments.nan_ranksite13.resources import below_normal, driver_sample
    below_normal(); seal = verify(); cfg = read(ROOT / 'config.json')
    decision = read(RUNS / 'eligibility.json'); assert decision['passed']
    for stage, hashes in decision['audit_sha256'].items():
        for b, h in hashes.items():
            assert sha(f'runs/{stage}/audits/{b}.json') == h
    import numpy as np
    import torch
    from experiments.nan_bpd01.backend import load_backend
    from experiments.nan_comp02.head import OperatorHead
    from experiments.nan_comp03.prompt import aliases
    from experiments.nan_gym05.common import prompt_ids
    from experiments.nan_fit07.worker import token_record, BudgetReached
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
    from mechinterp.nanbeige_bpd_gym.native_components import load_bpd
    from mechinterp.nanbeige_bpd_gym.recomputed_native import rollout_adapter
    from mechinterp.nanbeige_bpd_gym.resident_head import mount, retain_decoder_layers, check_parity
    from .fit import fit_one
    assert subprocess.check_output(['git', '-C', a.compactification_root, 'rev-parse', 'HEAD'], text=True).strip() == seal['adapter_commit']
    out = RUNS / a.backend; out.mkdir(exist_ok=False)
    os.environ['HF_HOME'] = str((out / 'HF_cache').resolve())
    torch.set_num_threads(1); torch.manual_seed(cfg['source_seed'])
    torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); cells = []; failure = None; status = 'failed'

    def event(name, **fields):
        v = dict(event=name, **fields)
        with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(v) + '\n')
        print(v, flush=True)

    def budget():
        if time.monotonic() - started > cfg['maximum_backend_seconds']:
            raise BudgetReached('Registered FIT16 between-stage resource budget reached; no retry')

    try:
        assert_oracle_isolated()
        loop, legacy, tok, _ = load_backend(a.compactification_root, a.snapshot, a.backend, chunk_rows=cfg['vocabulary_chunk_rows'])
        if a.backend == 'bf16':
            retain_decoder_layers(loop, cfg['BF16_resident_decoder_layers'])
        if loop.stage_layer is not None:
            loop.stage_layer.requires_grad_(False).eval()
        cpu = OperatorHead(legacy.weight, aliases(tok), chunk_rows=cfg['vocabulary_chunk_rows'])
        head = mount(loop, cpu); del legacy
        before = target_digest(loop); framework = load_bpd(a.bpd_root)
        write(out / 'manifest.json', dict(backend=a.backend, pid=os.getpid(), source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'),
            target_parameter_sha256=before, host_gold_parsed=False, CPU_threads=torch.get_num_threads(),
            decoder_GPU_layers=loop.n_gpu_layers, head_device=str(head.weight.device), embedding_device=str(loop.base.embed_tokens.weight.device)))
        write(out / 'driver_after_load.json', driver_sample())
        public = rows(ROOT / 'fixtures/public.jsonl'); sequences = {}; records = []; parity = []
        native = rollout_adapter(loop, None, {cfg['site']: cfg['rank']}, recompute=False)
        for i, row in enumerate(public):
            budget(); ids = prompt_ids(tok, a.snapshot, row['prompt']); sequences[row['id']] = ids
            upstream = Path('runs/nan_comp14') / a.backend / f"case_{row['upstream_ordinal']:04d}.json"
            old = read(upstream); assert old['id'] == row['id'] and ids == old['initial_prompt_ids']
            with torch.no_grad():
                logits, boundaries, _, _ = native(ids, 2, 0, None); phases = []
                for t in (0, 1):
                    phase = token_record(logits[t], tok); value = logits[t].flatten(); score = old['initial_phases'][t]['score']
                    assert phase['free_token_id'] == score['free_top_token_id'] and float(value.max()) == score['free_top_token_logit']
                    assert abs(phase['full_log_normalizer'] - score['full_log_normalizer']) <= 1e-9
                    if i == 0:
                        parity.append(check_parity(cpu, head, boundaries[t]))
                    phases.append(dict(pass_index_zero_based=t, **phase))
                record = dict(id=row['id'], root=row['root'], split=row['split'], world=row['world'], arm=row['arm'],
                    phases=phases, initial_prompt_ids_sha256=old['initial_prompt_ids_sha256'], upstream_case_sha256=sha(upstream))
                write(out / f'baseline_{i:03d}.json', record); records.append(record)
            del logits, boundaries, value; event('baseline_case_closed', ordinal=i)
        del cpu; gc.collect()
        write(out / 'baseline.json', dict(records=records, target_parameter_sha256=before, full_head_parity=parity,
            host_gold_parsed=False, contexts=len(records), source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json')))
        event('baseline_closed', contexts=len(records))
        bank_path = Path('runs/nan_gym05') / a.backend / 'linear_baselines/layer21_down_proj.npz'
        assert sha(bank_path) == seal['initialization'][a.backend]['file_sha256']
        with np.load(bank_path, allow_pickle=False) as z:
            initial = {k: z[k].copy() for k in ('V8', 'U8')}
        source = dict(BPD_root=a.bpd_root, manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'),
            initial_svd_file_sha256=sha(bank_path), effective_operator_sha256=seal['initialization'][a.backend]['effective_operator_sha256'])
        for passes in cfg['fit_passes']:
            budget()
            # Each depth starts from the same SVD and CI seed, without warm starts.
            cell = fit_one(loop, tok, public, sequences, cfg, a.backend, passes, out, initial, source, framework, budget, before)
            cells.append(dict(passes=passes, status=cell['status'], optimizer_steps_completed=cell['optimizer_steps_completed']))
            event('fit_condition_closed', **cells[-1]); gc.collect(); torch.cuda.empty_cache()
        assert target_digest(loop) == before; assert_oracle_isolated()
        status = 'completed' if all(c['status'] == 'completed' for c in cells) else 'completed_with_failed_conditions'
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error)); traceback.print_exc()
    finally:
        write(out / 'driver_terminal.json', driver_sample())
        write(out / 'receipt.json', dict(backend=a.backend, status=status, failure=failure, cells=cells,
            source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), elapsed_seconds=time.monotonic() - started,
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            target_parameter_sha256_unchanged=before if status.startswith('completed') else None,
            BPD_donor_transfer_attempted=False, sealed_test_accessed=False))
        event('worker_receipt_closed', status=status)
    if status == 'failed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
