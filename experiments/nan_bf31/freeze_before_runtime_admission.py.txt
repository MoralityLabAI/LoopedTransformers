"""Publish complete BF31 bindings before queue admission or pretrained inference."""
import ast
from datetime import datetime,timezone
from pathlib import Path
import subprocess
from .common import ROOT,RUNS,read,write,sha,load_sources,intervention_rows


def main():
    assert not RUNS.exists(),'No BF31 runtime may precede registration'
    from experiments.nan_ram28.common import verify as ram_verify
    from experiments.nan_storyaudit30.common import verify as audit_verify
    ram=ram_verify();audit_verify();public,pins=load_sources()
    assert len(public)==1549 and len(intervention_rows(public))==72
    cfg=read(ROOT/'config.json');assert cfg['backends']==['bf16'] and cfg['formal_inferential_comparisons']==0
    assert pins==ram['source_pins']
    token=read('experiments/nan_full26/tokenizer_receipt.json')['contexts']
    assert max(r['tokens'] for r in token.values())+256<=9216
    assert max(token[r['id']]['tokens'] for r in intervention_rows(public))+256<=4608
    proof=read(ROOT/'ENGINEERING.json');assert proof['status']=='passed' and proof['MOCK_ONLY'] and proof['CPU_contracts']==24
    assert proof['pretrained_forwards']==0 and not proof['CUDA_initialized']
    for name,digest in proof['source_sha256'].items():assert sha(ROOT/name)==digest,name
    extra={}
    for package in ('nan_ram28','nan_storyaudit30'):
        parent=Path('experiments')/package;seal=read(parent/'SOURCE_MANIFEST.json')
        extra[str(parent/'SOURCE_MANIFEST.json')]=sha(parent/'SOURCE_MANIFEST.json')
        for name,digest in seal['critical_files_sha256'].items():extra[str(parent/name)]=digest
    for name in ('lazy_bf16_storage.py','lazy_bf16_loop.py','lazy_bf16_backend.py','lazy_bf16_integration.py','bounded_cpu_engineering.py'):
        p=Path('mechinterp/nanbeige_bpd_gym')/name;extra[p.as_posix()]=sha(p)
    for name in ('lazy_bf16_storage_engineering.json','lazy_bf16_loop_integration.json','lazy_bf16_backend_integration.json'):
        p=Path('reports/nanbeige_bpd_gym')/name;extra[p.as_posix()]=sha(p)
    workflow='.github/workflows/lazy-bf16-storage-cpu.yml';extra[workflow]=sha(workflow)
    # Bind the actual local framework Python files in addition to the commit.
    bpd=Path('../BPD').resolve()
    assert subprocess.check_output(['git','-C',str(bpd),'rev-parse','HEAD'],text=True).strip()==ram['BPD_commit']
    framework=read('mechinterp/nanbeige_bpd_gym/BPD_FRAMEWORK_LOCK.json')['git_LF_files_sha256']
    for name,expected in framework.items():
        p=bpd/name
        assert __import__('hashlib').sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected
        extra[p.as_posix()]=sha(p)
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    critical={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.name!='SOURCE_MANIFEST.json'}
    write(ROOT/'SOURCE_MANIFEST.json',dict(experiment='NAN-BF-31',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        pretrained_inference_authorized_after_predecessors_end=True,critical_files_sha256=critical,additional_dependencies_sha256=extra,
        method_source_manifest_sha256=sha('experiments/nan_mlp27/SOURCE_MANIFEST.json'),
        RAM28_source_manifest_sha256=sha('experiments/nan_ram28/SOURCE_MANIFEST.json'),source_pins=pins,
        adapter_commit=ram['adapter_commit'],BPD_commit=ram['BPD_commit'],EvalMath_commit=ram['EvalMath_commit'],
        checkpoint_revision=cfg['checkpoint_revision'],baseline_contexts_per_backend=1549,AB_pairs_per_backend=216,
        distinct_root_ids=len({r['root'] for r in public}),matched_NF4_condition='NAN-RAM-28',NF4_relabelled=False,
        new_pretrained_started=False,pretrained_forwards_before_registration=0,task_score_selection=False,
        exposure='Earlier development results exposed; RAM28 NF4 in progress, only closed-journal coverage and resource metadata observed here. No new BF31 neural outcome or RAM28 generated-output grading selected tasks or criteria.',
        old_conditional_MLP27_qualification_claimed=False,original_failures_preserved=True,formal_inferential_comparisons=0,sealed_test_accessed=False))
    from .common import verify
    verify();print(dict(status='frozen_before_inference',manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),critical_files=len(critical),extra_bindings=len(extra)))


if __name__=='__main__':main()
