"""Future BF16 loader: immutable source spans and four resident GPU layers.

This entry point refuses to construct a model until its complete source
manifest exists and verifies. No worker/supervisor is started by importing it.
"""
EXPECTED_TARGET='cf3db880c052137e7ff7a0dac1e498174f78e32fd3ae9a384aca341fb1d09186'


def checkpoint_entry(compact_root,snapshot):
    import os,sys
    from pathlib import Path
    os.environ['ARG02_SNAPSHOT_NANBEIGE']=str(Path(snapshot).resolve())
    sys.path.insert(0,str(Path(compact_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
    entry=load_lock()['nanbeige']
    assert entry['revision']=='451ed48c3273ecef7ea8faaa43c31ce529763bb1'
    return entry


def verify_checkpoint(snapshot,entry):
    from pathlib import Path
    from .common import sha
    for name,digest in entry['files_sha256'].items():assert sha(Path(snapshot)/name)==digest,name


def load_backend(compact_root,snapshot,backend,*,device='cuda:0',chunk_rows=8192):
    from .common import ROOT,RUNS,read,write,sha,verify
    seal=verify()  # Must precede Torch import, CUDA allocation or model creation.
    from .admission import verify_worker
    verify_worker()  # Direct/manual calls cannot bypass the one-model queue.
    assert backend=='bf16','This separate control does not change any NF4 condition'
    import torch
    from pathlib import Path
    from experiments.nan_ram28.loader import empty_model
    from experiments.nan_ram28.resources import memory,soft_guard
    from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_storage import CheckpointStore,logical_target_digest
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_backend import build_loop
    from experiments.nan_bpd01.head import ChunkedHead
    if torch.device(device).type!='cuda':raise ValueError('The prospective model condition requires CUDA')
    NB.assert_oracle_isolated();cfg=read(ROOT/'config.json');entry=checkpoint_entry(compact_root,snapshot)
    verify_checkpoint(snapshot,entry)
    from research.algernon.algebraic_reasoning_gym.arg02 import runner,nanbeige
    model=empty_model(snapshot);nanbeige.check_supported(model.config);store=CheckpointStore(snapshot)
    loop=build_loop(model,store,device,n_resident=cfg['BF16_resident_decoder_layers'],reference_modules=(cfg['learned_site'],),
        guard=lambda:soft_guard(cfg))
    from .stage_audit import audit
    stage_check=audit(loop,lambda:soft_guard(cfg));write(RUNS/'stage_copy_attestation.json',stage_check)
    digest=logical_target_digest(model,store);assert digest==EXPECTED_TARGET,'Native BF16 logical/resident parameter bytes changed'
    tok=nanbeige.load_tokenizer(Path(snapshot),entry['bos_token_id'])
    head=ChunkedHead(model.lm_head.weight,runner.op_token_ids(tok),device=device,chunk_rows=chunk_rows)
    write(RUNS/'loader_attestation.json',dict(backend='bf16',source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        logical_and_resident_parameter_sha256=digest,reference_parameter_sha256=EXPECTED_TARGET,
        checkpoint_files_verified=True,stage_copy_attestation_sha256=sha(RUNS/'stage_copy_attestation.json'),
        full_CPU_decoder_materialized=False,storage_layout=loop.storage_layout,
        no_pretrained_forward_yet=True,head_device_before_mount=str(head.weight.device),memory=memory(),
        source_parameter_bytes=sum(v.length for v in store.spans.values()),sealed_test_accessed=False))
    NB.assert_oracle_isolated()
    return loop,head,tok,entry
