"""Real native gradients, fixed SVD factors and zero optimizer steps."""
import argparse
import gc
import hashlib
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from .common import ROOT, RUNS, read, sha, verify, write


def main():
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--compactification-root', required=True)
    p.add_argument('--bpd-root', required=True); a = p.parse_args()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ.pop('TRANSFORMERS_CACHE', None)
    from experiments.nan_ranksite13.resources import below_normal, driver_sample
    below_normal(); seal = verify(); cfg = read(ROOT / 'config.json')
    qualification = read(RUNS / 'eligibility.json'); assert qualification['passed']
    assert qualification['upstream_comp14_seal'] == cfg['upstream_comp14_seal']
    for b, expected in qualification['audit_sha256'].items():
        assert sha(Path('runs/nan_comp14/audits') / (b + '.json')) == expected
    import numpy as np
    import torch
    from experiments.nan_bpd01.backend import load_backend
    from experiments.nan_comp02.head import OperatorHead
    from experiments.nan_comp03.prompt import aliases
    from experiments.nan_gym05.common import prompt_ids
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
    from mechinterp.nanbeige_bpd_gym.native_components import load_bpd
    from mechinterp.nanbeige_bpd_gym.recomputed_native import attach_resident_components, ResidentSavedTensors, rollout_adapter
    from mechinterp.nanbeige_bpd_gym.resident_head import mount, retain_decoder_layers, check_parity
    assert subprocess.check_output(['git', '-C', a.compactification_root, 'rev-parse', 'HEAD'], text=True).strip() == seal['adapter_commit']
    out = RUNS / a.backend; out.mkdir(exist_ok=False)
    os.environ['HF_HOME'] = str((out / 'HF_cache').resolve())
    torch.set_num_threads(1); torch.manual_seed(cfg['source_seed'])
    torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']); torch.cuda.reset_peak_memory_stats()
    source = read(ROOT / 'fixtures/public.json'); started = time.monotonic(); conditions = []; failure = None; status = 'failed'

    def event(name, **fields):
        import json
        row = dict(event=name, **fields)
        with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(row, flush=True)

    def budget():
        if time.monotonic() - started > cfg['maximum_backend_seconds']:
            raise TimeoutError('Registered between-stage gradient budget reached; no retry')

    def save(name, arrays):
        path = out / name
        with path.open('xb') as f:
            np.savez_compressed(f, **arrays)
        return dict(file=name, sha256=sha(path))

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
            target_parameter_sha256=before, optimizer_steps=0, host_gold_parsed=False, CPU_threads=torch.get_num_threads(),
            decoder_GPU_layers=loop.n_gpu_layers, head_device=str(head.weight.device), embedding_device=str(loop.base.embed_tokens.weight.device)))
        write(out / 'driver_after_load.json', driver_sample())
        bank_path = Path('runs/nan_gym05') / a.backend / 'linear_baselines/layer21_down_proj.npz'
        assert sha(bank_path) == seal['initialization'][a.backend]['file_sha256']
        with np.load(bank_path, allow_pickle=False) as z:
            initial = dict(V=z['V8'].copy(), U=z['U8'].copy())
        baseline = {}; sequences = {}; parity = []
        native = rollout_adapter(loop, None, {cfg['site']: cfg['rank']}, recompute=False)
        for i, row in enumerate(source):
            budget(); ids = prompt_ids(tok, a.snapshot, row['prompt']); sequences[row['id']] = ids
            old = read(Path('runs/nan_comp14') / a.backend / f"case_{row['upstream_ordinal']:04d}.json")
            assert old['id'] == row['id'] and ids == old['initial_prompt_ids']
            with torch.no_grad():
                logits, boundaries, _, _ = native(ids, 2, 0, None)
                for t in (0, 1):
                    score = old['initial_phases'][t]['score']; value = logits[t].flatten()
                    assert int(value.argmax()) == score['free_top_token_id']
                    assert float(value.max()) == score['free_top_token_logit']
                    assert abs(float(torch.logsumexp(value.double(), 0)) - score['full_log_normalizer']) <= 1e-9
                    if i == 0:
                        parity.append(check_parity(cpu, head, boundaries[t]))
                baseline[row['id']] = dict(logits={t: x.cpu().clone() for t, x in logits.items()},
                    boundaries={t: x.cpu().clone() for t, x in boundaries.items()}, upstream_case_sha256=sha(Path('runs/nan_comp14') / a.backend / f"case_{row['upstream_ordinal']:04d}.json"))
                archived = {f'logits_{t}': logits[t].flatten().cpu().numpy() for t in (0, 1)}
                archived.update({f'boundary_{t}': boundaries[t].flatten().float().cpu().numpy() for t in (0, 1)})
                write(out / f'baseline_{i:02d}.json', dict(source_id=row['id'], upstream_ordinal=row['upstream_ordinal'],
                    upstream_case_sha256=baseline[row['id']]['upstream_case_sha256'],
                    archive=save(f'baseline_{i:02d}.npz', archived), input_ids_sha256=old['initial_prompt_ids_sha256']))
                del archived
            del logits, boundaries, value
        del cpu; gc.collect(); write(out / 'head_parity.json', dict(checks=parity, upstream_baseline_prefixes_verified=4))
        event('baseline_closed', prefixes=4)
        for passes in cfg['passes']:
            if conditions and conditions[-1]['status'] != 'passed':
                conditions.append(dict(passes=passes, status='unrun_T1_prerequisite_failed', cases=0)); continue
            budget(); instrument = attach_resident_components(loop, {cfg['site']: cfg['rank']}, bpd_root=a.bpd_root,
                ablation=framework['config'].AblationConfig(mode='zero', sphere_rescale=False), max_passes=passes)
            core = instrument.wrappers[cfg['site']]
            with torch.no_grad():
                core.V.copy_(torch.from_numpy(initial['V']).to(loop.device)); core.U.copy_(torch.from_numpy(initial['U']).to(loop.device))
            assert (core.d_in, core.d_out) == (cfg['input_width'], cfg['output_width'])
            effective = core.W_target.detach().float().cpu().contiguous().numpy()
            assert hashlib.sha256(memoryview(effective)).hexdigest() == seal['initialization'][a.backend]['effective_operator_sha256']; del effective
            condition = dict(passes=passes, status='passed', cases=0)
            owner = adapter = None
            try:
                for i, row in enumerate(source):
                    budget(); arrays = {}; record = dict(passes=passes, source_index=i, source_id=row['id'], status='failed')
                    owner = ResidentSavedTensors(loop, cfg['maximum_added_saved_tensor_CPU_bytes'], constants=[core.W_target])
                    adapter = rollout_adapter(loop, owner, {cfg['site']: cfg['rank']}, chunk_rows=cfg['vocabulary_chunk_rows'])
                    clean = baseline[row['id']]
                    teacher = framework['components'].RolloutTrajectory(
                        {t: clean['logits'][t].to(loop.device) for t in range(passes)},
                        {t: clean['boundaries'][t].to(loop.device) for t in range(passes)}, {}, None, None)
                    schedule = framework['components'].PassMaskSchedule; length = len(sequences[row['id']])
                    loss = pred = prediction = states = all_logits = all_states = None
                    try:
                        core.V.grad = None; core.U.grad = None
                        with instrument.masked_components({cfg['site']: schedule(torch.ones((1, passes, length, cfg['rank']), device=loop.device), torch.ones((1, passes, length), device=loop.device))}):
                            all_logits, all_states, _, _ = adapter(sequences[row['id']], passes, 0, None)
                        assert all(torch.equal(all_logits[t], teacher.readouts[t]) and torch.equal(all_states[t], teacher.boundary_states[t]) for t in range(passes))
                        for t in range(passes):
                            arrays[f'clean_logits_{t}'] = teacher.readouts[t].flatten().cpu().numpy()
                            arrays[f'all_one_logits_{t}'] = all_logits[t].detach().flatten().cpu().numpy()
                            arrays[f'clean_boundary_{t}'] = teacher.boundary_states[t].flatten().float().cpu().numpy()
                            arrays[f'all_one_boundary_{t}'] = all_states[t].detach().flatten().float().cpu().numpy()
                        del all_logits, all_states; all_logits = all_states = None; gc.collect()
                        with owner.context():
                            with instrument.masked_components({cfg['site']: schedule(
                                torch.full((1, passes, length, cfg['rank']), cfg['component_mask'], device=loop.device),
                                torch.full((1, passes, length), cfg['residual_mask'], device=loop.device))}):
                                prediction, states, _, _ = adapter(sequences[row['id']], passes, 0, None)
                            pred = framework['components'].RolloutTrajectory(prediction, states, {}, None, None)
                            loss = framework['losses'].trajectory_recon_loss(pred, teacher, readout_policy='readout_only')[0]
                            loss.backward()
                        assert math.isfinite(float(loss.detach())) and float(loss.detach()) > 0
                        norms = {}
                        for name in ('V', 'U'):
                            gradient = getattr(core, name).grad
                            assert gradient is not None and torch.isfinite(gradient).all() and torch.isfinite(gradient.norm()) and bool(gradient.norm() > 0)
                            arrays[name + '_gradient'] = gradient.detach().cpu().numpy().copy(); norms[name] = float(gradient.norm())
                        for t in range(passes):
                            arrays[f'masked_logits_{t}'] = prediction[t].detach().flatten().cpu().numpy()
                            arrays[f'masked_boundary_{t}'] = states[t].detach().flatten().float().cpu().numpy()
                        record.update(status='passed', loss=float(loss.detach()), gradient_L2=norms, all_one_closure_bit_exact=True,
                            upstream_case_sha256=clean['upstream_case_sha256'])
                    except Exception as error:
                        record.update(failure_type=type(error).__name__, failure_message=str(error)); traceback.print_exc(); condition['status'] = 'failed'
                    finally:
                        record['archive'] = save(f't{passes}_case_{i:02d}.npz', arrays)
                        del loss, pred, prediction, states, all_logits, all_states, teacher; gc.collect()
                        record['saved_tensor_statistics'] = owner.statistics()
                        record['CUDA_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                        record['CUDA_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                        write(out / f't{passes}_case_{i:02d}.json', record)
                        condition['cases'] += 1; event('gradient_case_closed', passes=passes, source_index=i, status=record['status'])
                    if record['status'] != 'passed':
                        break
                assert torch.equal(core.V.detach().cpu(), torch.from_numpy(initial['V'])) and torch.equal(core.U.detach().cpu(), torch.from_numpy(initial['U']))
            finally:
                instrument.close(); core.V.grad = None; core.U.grad = None; del core, instrument, owner, adapter; gc.collect(); torch.cuda.empty_cache()
            conditions.append(condition); event('condition_closed', **condition)
        assert target_digest(loop) == before; assert_oracle_isolated(); status = 'completed'
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error)); traceback.print_exc()
    finally:
        write(out / 'driver_terminal.json', driver_sample())
        write(out / 'receipt.json', dict(backend=a.backend, status=status, failure=failure, conditions=conditions,
            source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), elapsed_seconds=time.monotonic() - started,
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            target_parameter_sha256_unchanged=before if status == 'completed' else None,
            runtime=dict(torch=torch.__version__, CUDA=torch.version.cuda),
            optimizer_steps=0, factors_trained=False, BPD_donor_transfer_attempted=False, sealed_test_accessed=False))
        event('worker_receipt_closed', status=status)
    if status != 'completed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
