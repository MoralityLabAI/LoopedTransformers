"""New cached backend assembly, refused before Torch until own queue admission."""
TARGETS=dict(bf16='cf3db880c052137e7ff7a0dac1e498174f78e32fd3ae9a384aca341fb1d09186',
    nf4='8e80e7d748d5daf858a0b37dd7843d38952554ce3aabea1104e52cf1ef7663b3')


def load(a,cfg):
    from .common import verify,verify_worker,write,RUNS
    verify('model');verify_worker(a.backend)
    import torch
    from pathlib import Path
    from experiments.nan_bf31.loader import checkpoint_entry,verify_checkpoint
    from experiments.nan_ram28.loader import empty_model,materialize
    from experiments.nan_ram28.resources import soft_guard
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import _reinit_rotary_buffers,assert_oracle_isolated
    from mechinterp.nanbeige_bpd_gym.chunked_backend import build_nf4_loop
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_backend import build_loop
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_storage import CheckpointStore
    from experiments.nan_recomp09.common import target_digest
    assert_oracle_isolated();entry=checkpoint_entry(a.compactification_root,a.snapshot);verify_checkpoint(a.snapshot,entry)
    from research.algernon.algebraic_reasoning_gym.arg02 import nanbeige
    model=empty_model(a.snapshot);nanbeige.check_supported(model.config);store=None;stage=None
    if a.backend=='nf4':
        mapped=materialize(model,a.snapshot)
        for layer in model.model.layers:_reinit_rotary_buffers(layer,torch.device('cpu'))
        model.eval().requires_grad_(False)
        loop=build_nf4_loop(model,'cuda:0',guard=lambda:soft_guard(cfg))
        layout=dict(**mapped,resident_decoder_layers=22,complete_BF16_CPU_heap=False,head_device='cpu')
    else:
        store=CheckpointStore(a.snapshot)
        loop=build_loop(model,store,'cuda:0',n_resident=cfg['BF16_resident_decoder_layers'],reference_modules=(cfg['learned_site'],),guard=lambda:soft_guard(cfg))
        from experiments.nan_bf31.stage_audit import audit
        stage=audit(loop,lambda:soft_guard(cfg));layout=loop.storage_layout
    before=target_digest(loop);assert before==TARGETS[a.backend],'Pinned target bytes differ before first decoder forward'
    tok=nanbeige.load_tokenizer(Path(a.snapshot),entry['bos_token_id'])
    write(RUNS/a.backend/'loader_attestation.json',dict(backend=a.backend,target_parameter_sha256=before,
        storage_layout=layout,actual_stage_checks=stage,no_decoder_forward_yet=True,sealed_test_accessed=False))
    assert_oracle_isolated();return loop,tok,entry,before
