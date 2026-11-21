"""Freeze source and administrative selectors, without grading neural outputs."""
from datetime import datetime, timezone
from .common import ROOT, OLD, OLD_RUNS, read, rows, sha, write, select_prefix


def main():
    outcome = 'experiments/nan_ram28/RESOURCE_OUTCOME.json'
    administrative = [outcome, str(OLD_RUNS/'queue_events.jsonl'), str(OLD_RUNS/'bf16/events.jsonl'),
        str(OLD_RUNS/'bf16/manifest.json'), str(OLD_RUNS/'bf16/head_parity.json'),
        str(OLD_RUNS/'nf4/receipt.json'), str(OLD_RUNS/'audits/nf4.json')]
    assert sha(OLD/'SOURCE_MANIFEST.json')=='d3d0587416d0bf92e7b5d044f527d1fc6bbb630c6d3f074e5fcf0d1e03b6f4bc'
    count = select_prefix(rows(OLD_RUNS/'bf16/events.jsonl'), rows(OLD_RUNS/'queue_events.jsonl'),
                          read(outcome), (OLD_RUNS/'bf16/receipt.json').exists())
    write(ROOT/'registration.json', dict(experiment='NAN-CTX21-RESOURCE29', frozen_UTC=datetime.now(timezone.utc).isoformat(),
        closed_prefix_count=count, BF16_target_parameter_sha256_before=read(OLD_RUNS/'bf16/manifest.json')['target_parameter_sha256'],
        administrative_artifacts_sha256={name:sha(name) for name in administrative},
        minimum_start_available_RAM_bytes=5*2**30, minimum_available_RAM_bytes=4*2**30, maximum_private_bytes=3*2**28,
        original_terminal_gate_changed=False, original_receipt_synthesized=False, retrospective_administrative_registration=True,
        pretrained_forwards=0, sealed_test_accessed=False))
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SOURCE_MANIFEST.json')
    dependencies = ['experiments/nan_ram28/resources.py', 'experiments/nan_ranksite13/resources.py']
    write(ROOT/'SOURCE_MANIFEST.json', dict(experiment='NAN-CTX21-RESOURCE29', frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        additional_dependencies_sha256={name:sha(name) for name in dependencies},
        original_source_manifest_sha256=sha(OLD/'SOURCE_MANIFEST.json'),
        pregrading_source_freeze=True, preinference_freeze=False, original_gates_unchanged=True,
        no_Torch_model_CUDA=True, pretrained_forwards=0, sealed_test_accessed=False))
    print(dict(seal=sha(ROOT/'SOURCE_MANIFEST.json'), files=len(files), closed_prefix=count))


if __name__=='__main__': main()
