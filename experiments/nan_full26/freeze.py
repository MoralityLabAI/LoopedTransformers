"""Freeze all new code, source and qualified fixed-asset bindings before inference."""
from datetime import datetime,timezone
from pathlib import Path
from .common import ROOT,RUNS,read,sha,write,load_sources

def main():
    assert not RUNS.exists()
    public,pins=load_sources();token=read(ROOT/'tokenizer_receipt.json');assert token['source_pins']==pins
    for name in ('engineering_receipt.json','generation_receipt.json'):
        r=read(ROOT/name);assert r['status']=='passed' and r['pretrained_model_calls']==0
        for path,h in r['source_sha256'].items():assert sha(ROOT/path)==h
    files={}
    for root in [ROOT,Path('experiments/nan_story23'),Path('experiments/nan_bpd01'),Path('experiments/nan_comp02'),
        Path('experiments/nan_comp03'),Path('experiments/nan_gym05'),Path('experiments/nan_recomp09'),
        Path('experiments/nan_fit07'),Path('experiments/nan_ranksite13'),Path('experiments/nan_xref19'),
        Path('mechinterp/nanbeige_bpd_gym')]:
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.jsonl','.md'):
                if p==ROOT/'SOURCE_MANIFEST.json':continue
                files[p.as_posix()]=sha(p)
    files[ (ROOT/'.gitattributes').as_posix()]=sha(ROOT/'.gitattributes')
    for p in ('mechinterp/dark_fiber_gyms/arg03_nanbeige.py','mechinterp/dark_fiber_gyms/q4_3050.py',
        'mechinterp/sae_causal_transfer/adapter.py','runs/nan_gym05/cohort/host_labels.jsonl',
        '.github/workflows/nan-full26-cpu.yml'):
        files[p]=sha(p)
    for stage in ('nan_comp14','nan_ctx08','nan_ctx21'):
        for name in ('public.jsonl','host_labels.jsonl'):
            p=f'experiments/{stage}/fixtures/{name}';files[p]=sha(p)
    inventory=read('experiments/nan_xref19/CACHE_INVENTORY.json')['backends']
    from experiments.nan_story23.common import fit_asset
    for backend in ('nf4','bf16'):
        asset=fit_asset(backend,2);assert asset
        for p in (asset['checkpoint'],str(Path(asset['folder'])/'receipt.json'),str(Path(asset['folder'])/'diagnostic_local.json'),f'runs/nan_fit16_io18/audits/{backend}.json'):
            files[p]=sha(p)
        for names in inventory[backend]['dictionaries'].values():
            for description in names.values():files[description['path']]=description['sha256']
    write(ROOT/'SOURCE_MANIFEST.json',dict(experiment='NAN-FULL-26',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        source_pins=pins,files=files,baseline_contexts_per_backend=1549,AB_pairs_per_backend=216,
        independent_roots=len({r['root'] for r in public}),checkpoint_revision=read(ROOT/'config.json')['checkpoint_revision'],
        adapter_commit='1ac36353b36581b6a634626bcd55542e53dae456',BPD_commit='8f658b789138d73f497b1a2a05f5783745147d40',
        EvalMath_commit='accf4d9040bee8daeabc6b07003bb484708996c0',
        exposure='Prior development results exposed; CTX21 NF4 currently running; no FULL26 pretrained forward.',
        pretrained_started=False,formal_inferential_comparisons=0,sealed_test_accessed=False))
    print(dict(seal_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),bindings=len(files),baseline=1549,AB_pairs=216))

if __name__=='__main__':main()
