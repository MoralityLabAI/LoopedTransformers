"""Prospective complete source/dependency freeze; no checkpoint tensor loading."""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import subprocess
from .common import ROOT,RUNS,read,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);a=p.parse_args()
    assert not any((RUNS/b).exists() for b in ('nf4','bf16')) and not (RUNS/'operations').exists()
    receipts={}
    for name in ('independent_source_audit.json','numeric_engineering_v4.json','generation_engineering_v1.json','linux_discovery_v1.json','mock_transport_v2.json'):
        value=read(ROOT/'fixtures'/name);assert value['status']=='passed'
        for file,h in value.get('source_sha256',{}).items():assert sha(ROOT/file)==h,(name,file)
        if 'entry_point_sha256' in value:assert sha('nanbeige_storyworld_suite.py')==value['entry_point_sha256']
        receipts[name]=sha(ROOT/'fixtures'/name)
    assert read(ROOT/'fixtures/mock_transport_v2.json')['official_mock_episodes']==6
    assert read(ROOT/'fixtures/numeric_engineering_v4.json')['all_one_native_bit_exact']
    tokenizer=read(ROOT/'fixtures/tokenizer_receipt.json')
    assert tokenizer['public_sha256']==sha(ROOT/'fixtures/public.jsonl') and tokenizer['maximum_canonical_trace_tokens']<=64
    for name,h in tokenizer['tokenizer_files_sha256'].items():assert sha(Path(a.snapshot)/name)==h
    for kind in ('public','host'):
        assert read(ROOT/'fixtures/independent_source_audit.json')[kind+'_sha256']==sha(ROOT/'fixtures'/(kind+'.jsonl' if kind=='public' else 'host_labels.jsonl'))
    dependencies=set()
    # Bind inherited loader, tokenizer, observer, physics, auditing and all
    # transitively declared dependencies of the previous registered sources.
    stages=('nan_bpd01','nan_comp02','nan_comp03','nan_gym05','nan_fit07','nan_recomp09','nan_ranksite13',
        'nan_comp14','nan_grad15','nan_fit16','nan_fit16_io18','nan_xref19','nan_prime06','nan_prime20')
    for stage in stages:
        folder=Path('experiments')/stage
        dependencies.update(str(f).replace('\\','/') for f in folder.rglob('*')
            if f.is_file() and f.suffix in ('.py','.json') and '__pycache__' not in f.parts)
        manifest=folder/'SOURCE_MANIFEST.json'
        if manifest.exists():
            value=read(manifest)
            for key in ('dependencies','dependencies_sha256','observational_bridge_sha256'):
                dependencies.update(name for name in value.get(key,{}) if Path(name).is_file())
    dependencies.update(str(f).replace('\\','/') for f in Path('mechinterp/nanbeige_bpd_gym').glob('*.py'))
    dependencies.update({'mechinterp/nanbeige_bpd_gym/BPD_FRAMEWORK_LOCK.json','mechinterp/dark_fiber_gyms/arg03_nanbeige.py',
        'mechinterp/sae_causal_transfer/adapter.py','nanbeige_storyworld_suite.py','.github/workflows/nan-story23-cpu.yml'})
    bindings={}
    for name in sorted(dependencies):
        if name in ('nanbeige_storyworld_suite.py','.github/workflows/nan-story23-cpu.yml'):
            blob=Path(name).read_bytes().replace(b'\r\n',b'\n')
        else:blob=subprocess.check_output(['git','show','HEAD:'+name])
        bindings[name]=dict(runtime_sha256=sha(name),git_sha256=hashlib.sha256(blob).hexdigest())
    critical={str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for f in sorted(ROOT.rglob('*'))
        if f.is_file() and f.suffix in ('.py','.json','.jsonl','.md','.sh') and f.name!='SOURCE_MANIFEST.json' and '__pycache__' not in f.parts}
    critical['.gitattributes']=sha(ROOT/'.gitattributes')
    write(ROOT/'SOURCE_MANIFEST.json',dict(experiment='NAN-STORY-23',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical,dependencies=bindings,engineering_receipts=receipts,
        checkpoint_revision=read(ROOT/'config.json')['checkpoint_revision'],tokenizer_files_sha256=tokenizer['tokenizer_files_sha256'],
        BPD_commit='8f658b789138d73f497b1a2a05f5783745147d40',EvalMath_commit='accf4d9040bee8daeabc6b07003bb484708996c0',
        adapter_commit='1ac36353b36581b6a634626bcd55542e53dae456',
        exposure='Prior source-only/native/SDK/SAE/fit development results exposed; IO18 BF16 closing diagnostics and later queues pending; no STORY23 checkpoint forward or source outcome filtering.',
        conditional_asset_policy='Only independently passed sixteen-step IO18 terminal fits; all CAL-only SAE dictionary bytes from pinned XREF19 inventory.',
        formal_inferential_comparisons=0,STORY23_pretrained_started=False,sealed_test_accessed=False))
    print(dict(source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),critical_files=len(critical),dependencies=len(bindings)))


if __name__=='__main__':main()
