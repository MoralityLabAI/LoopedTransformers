"""Random-only comparison of default and explicit-effective NF4 caching."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
import torch
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_gym05.observe import TowerObserver
from experiments.nan_fit07.io import sha, write
from .effective_nf4 import mount_effective_nf4
from .native_cached import NativeTwoLoopCache
from .resident_head import mount, retain_decoder_layers, full_logits
from .validate_native_trainer import parameter_digest


def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cache_dir = out / 'HF_cache'
    cache_dir.mkdir()
    os.environ['HF_HOME'] = str(cache_dir.resolve())
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.pop('TRANSFORMERS_CACHE', None)
    if os.name == 'nt':
        import ctypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        if not k.SetPriorityClass(k.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.05)
    started = time.monotonic()
    results = []
    baseline = {}
    nf_digest = None
    for policy in ('bf16', 'nf4_default', 'nf4_explicit_BF16'):
        torch.manual_seed(2026112011)
        loop = cached = observer = model = None
        result = dict(policy=policy, status='failed', rows=[])
        results.append(result)
        torch.cuda.reset_peak_memory_stats()
        try:
            model, _ = NB.build_nanbeige_model_and_loop(Path(a.snapshot), exec_device=torch.device('cpu'),
                dtype=torch.bfloat16, n_gpu_layers=22, random_init=True,
                config_overrides={'vocab_size':166144, 'num_hidden_layers':22, 'hidden_size':64,
                    'intermediate_size':128, 'num_attention_heads':4, 'num_key_value_heads':2,
                    'head_dim':16, 'kv_channels':16})
            loop = NativeLoop(model, 'cuda:0', quantized=policy != 'bf16')
            if policy == 'bf16':
                retain_decoder_layers(loop, 4)
            cpu = OperatorHead(loop.model.lm_head.weight, dict(I=[1], N=[2], S=[3], R=[4]), chunk_rows=8192)
            mount(loop, cpu)
            del cpu
            before = parameter_digest(loop.model)
            if policy == 'nf4_default':
                nf_digest = before
            if policy == 'nf4_explicit_BF16':
                if nf_digest is None or before != nf_digest:
                    raise AssertionError('The two NF4 policies did not receive identical packed parameters')
                replaced = mount_effective_nf4(loop)
                if parameter_digest(loop.model) != before:
                    raise AssertionError('Mounting the new condition changed parameter bytes or names')
                result['replaced_decoder_linears'] = len(replaced)
                result['packed_parameters_identical_to_default_NF4'] = True
            cached = NativeTwoLoopCache(loop)
            observer = TowerObserver(loop, (5, 11, 17, 21))
            rng = torch.Generator().manual_seed(2026112011)
            ids = torch.randint(1, 100, (19,), generator=rng).tolist()
            extra = torch.randint(1, 100, (8,), generator=rng).tolist()
            for step in range(9):
                if step:
                    ids.append(extra[step - 1])
                with torch.inference_mode():
                    torch.cuda.synchronize()
                    mark = time.monotonic()
                    two, observed, one = observer.forward({'context':ids}, capture=True)
                    reference = [one[0][:, -1:], two[0][:, -1:]]
                    expected = [full_logits(loop.model.lm_head.weight, h, loop.device, 8192) for h in reference]
                    torch.cuda.synchronize()
                    reference_seconds = time.monotonic() - mark
                    mark = time.monotonic()
                    states, tape = cached.prefill(ids, capture=True) if step == 0 else cached.step(extra[step - 1], capture=True)
                    actual = [full_logits(loop.model.lm_head.weight, h, loop.device, 8192) for h in states]
                    torch.cuda.synchronize()
                    cache_seconds = time.monotonic() - mark
                    if observed.keys() != tape.keys() or len(tape) != 40:
                        raise AssertionError('Incomplete physical observations')
                    physical_error = max(relative(tape[k], observed[k]) for k in tape)
                    for phase, (candidate, target, logits, target_logits) in enumerate(zip(states, reference, actual, expected), 1):
                        row = dict(step=step, loop=phase, boundary_relative_l2=relative(candidate, target),
                            full_vocabulary_logit_max_abs=float((logits - target_logits).abs().max()),
                            free_argmax_equal=int(logits.argmax()) == int(target_logits.argmax()),
                            logits_checked=len(logits), physical_tape_site_relative_l2_max=physical_error,
                            cache_bytes=cached.cache_bytes(), cache_sequence_length=cached.length,
                            prefill_boundary_bit_exact=torch.equal(candidate, target),
                            prefill_logits_bit_exact=torch.equal(logits, target_logits),
                            both_loops_reference_seconds=reference_seconds, both_loops_cached_seconds=cache_seconds)
                        result['rows'].append(row)
                        if step == 0 and not (row['prefill_boundary_bit_exact'] and row['prefill_logits_bit_exact']
                                and all(torch.equal(tape[k], observed[k]) for k in tape)):
                            raise AssertionError('Within-policy prefill changed')
                        if policy == 'nf4_default':
                            baseline[(step, phase)] = (target.float().cpu().clone(), target_logits.clone())
                        if policy == 'nf4_explicit_BF16':
                            old_boundary, old_logits = baseline[(step, phase)]
                            row['cross_policy_full_prefix_boundary_relative_l2'] = relative(target.cpu(), old_boundary)
                            row['cross_policy_full_prefix_logit_max_abs'] = float((target_logits - old_logits).abs().max())
                            row['cross_policy_full_prefix_argmax_equal'] = int(target_logits.argmax()) == int(old_logits.argmax())
                    del states, tape, two, observed, one, reference, candidate, target, expected, actual, logits, target_logits
            unchanged = parameter_digest(loop.model) == before
            if not unchanged:
                raise AssertionError('Target parameters changed')
            violations = [dict(step=r['step'], loop=r['loop']) for r in result['rows'] if r['step'] > 0
                and (r['boundary_relative_l2'] > .02 or r['full_vocabulary_logit_max_abs'] > .1)]
            result.update(status='passed' if not violations else 'failed_engineering_bound',
                engineering_bound_violations=violations, target_parameters_unchanged=True,
                target_parameter_sha256=before, physical_vectors_checked=9*40,
                independent_phase_cache_instances=True,
                incremental_free_argmax_agreements=sum(r['free_argmax_equal'] for r in result['rows'] if r['step'] > 0),
                incremental_readouts=16, CPU_threads=torch.get_num_threads())
        except Exception as error:
            result.update(failure_type=type(error).__name__, failure_message=str(error))
            traceback.print_exc()
        finally:
            if cached is not None:
                cached.close()
            if observer is not None:
                observer.close()
            result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            print({'policy':policy, 'status':result['status'], 'rows':len(result['rows'])}, flush=True)
            del cached, observer, loop, model
            gc.collect()
            torch.cuda.empty_cache()
    root = Path(__file__).parent
    receipt = dict(status='passed' if all(r['status'] == 'passed' for r in results) else 'failed', results=results,
        evidence_class='random precision/cache engineering; no pretrained or causal evidence',
        source_sha256={n:sha(root/n) for n in ('effective_nf4.py', 'validate_effective_nf4.py',
            'EFFECTIVE_NF4_ENGINEERING_PLAN.md', 'native_cached.py')},
        elapsed_seconds=time.monotonic()-started, pretrained_weights_loaded=False,
        sealed_test_accessed=False, BPD_transfer_attempted=False, formal_inferential_comparisons=0)
    write(out/'receipt.json', receipt)
    print({'status':receipt['status'], 'elapsed_seconds':receipt['elapsed_seconds']}, flush=True)
    if receipt['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
