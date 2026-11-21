"""Bind all 72 source slots before any generated-text/state inspection."""
from datetime import datetime,timezone
from .common import ROOT,read,write,sha
from .run import roster


def main():
    from experiments.nan_ram28.common import verify
    verify();public=roster()
    assert len(public)==72 and len({r['root'] for r in public})==12
    assert len({r['root'] for r in public if r['split']=='calibration'})==3
    assert len({r['root'] for r in public if r['split']=='development'})==9
    from experiments.nan_full26.common import load_sources
    assert [r['id'] for r in load_sources()[0][:72]]==[r['id'] for r in public]
    tokens=read('experiments/nan_full26/tokenizer_receipt.json')
    assert all(tokens['contexts'][r['id']]['tokens']+256<=4608 for r in public),'Story rows must all admit real B2'
    write(ROOT/'registration.json',dict(experiment='NAN-STORYAUDIT-30',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        ordered_source_ids=[r['id'] for r in public],registered_trajectories=72,source_roots=12,CAL_roots=3,DEV_roots=9,
        minimum_start_available_RAM_bytes=5*2**30,minimum_available_RAM_bytes=4*2**30,maximum_private_bytes=3*2**28,
        packages=dict(numpy='2.5.5',transformers='4.46.3',tokenizers='0.20.3'),
        CPU_threads=1,priority='BelowNormal',endpoint_selection_uses_model_outputs=False,
        freeze_before_early_grading=True,RAM28_inference_already_started=True,
        no_original_inference_or_host_method_changes=True,new_pretrained_forwards=0,sealed_test_accessed=False))
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SOURCE_MANIFEST.json')
    write(ROOT/'SOURCE_MANIFEST.json',dict(experiment='NAN-STORYAUDIT-30',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        RAM28_source_manifest_sha256=sha('experiments/nan_ram28/SOURCE_MANIFEST.json'),
        new_pretrained_forwards=0,sealed_test_accessed=False))
    print(dict(files=len(files),seal=sha(ROOT/'SOURCE_MANIFEST.json'),generated_text_or_states_inspected=False))


if __name__=='__main__':main()
