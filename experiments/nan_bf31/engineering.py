"""MOCK_ONLY: run the new separate-root host/geometry/report integration."""
import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from mechinterp.nanbeige_bpd_gym.bounded_cpu_engineering import CPUBudget
from .common import ROOT,read,write,append,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir)
    assert 'MOCK_ONLY' in out.name
    with CPUBudget(out) as budget:
        import torch
        torch.set_num_threads(1)
        suite=unittest.TestSuite()
        for name in ('mechinterp.nanbeige_bpd_gym.lazy_bf16_storage_tests','mechinterp.nanbeige_bpd_gym.lazy_bf16_loop_tests',
                     'experiments.nan_bf31.tests','experiments.nan_bf31.stage_tests'):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
        result=unittest.TextTestRunner(verbosity=1).run(suite);assert result.wasSuccessful() and result.testsRun==24
        from . import host,inputs,report,host_budget
        old=Path('runs/nan_mlp27_engineering/synthetic_host_v3');assert read(old/'host/receipt.json')['closed_episodes_audited']==22
        nf=out/'NF4_MOCK_ONLY';own=out/'BF16_MOCK_ONLY';nf.mkdir();own.mkdir();fixture=out/'source_MOCK_ONLY';fixture.mkdir()
        oldseal=sha('experiments/nan_ram28/SOURCE_MANIFEST.json')
        write(fixture/'SOURCE_MANIFEST.json',dict(MOCK_ONLY=True,RAM28_source_manifest_sha256=oldseal))
        shutil.copyfile(ROOT/'config.json',fixture/'config.json')
        nfhost=nf/'host';nfhost.mkdir();expected_prefixes=0;episodes=0
        for backend,dest in [('nf4',nf/'nf4'),('bf16',own/'bf16')]:
            dest.mkdir();source=old/backend;entries=[]
            import json
            for line in (source/'closed.jsonl').read_bytes().splitlines(keepends=True):
                if line.endswith(b'\n'):entries.append(json.loads(line))
            assert len(entries)==11
            closed={e['file']:read(source/e['file']) for e in entries}
            active={}
            for name,r in closed.items():
                stem=name.rsplit('_r',1)[0]
                for step in range(len(r['fragments'])):active.setdefault((stem,step),[]).append(name)
            for name,r in closed.items():
                stem=name.rsplit('_r',1)[0];fragments=[];prefix_keys=[]
                for step,f in enumerate(r['fragments']):
                    original=read(source/f['file']);pname=name.replace('_closed.json',f'_p{step:03d}.json');state=pname.replace('.json','.npz')
                    shutil.copyfile(source/original['state_file'],dest/state)
                    names=sorted(active[stem,step]);B=len(names);row=names.index(name)
                    original.update(state_file=state,state_sha256=sha(dest/state),source_id=r['source_id'],mode=r['mode'],step=step,
                        prefix_ids_sha256=__import__('hashlib').sha256(('MOCK-input-'+r['source_id']+str(step)).encode()).hexdigest(),
                        history_sha256=__import__('hashlib').sha256(('MOCK-history-'+r['source_id']+str(step)).encode()).hexdigest(),
                        batch_size=B,batch_row=row,padded_length=5+step,actual_length=5+step,padding_tokens=0)
                    write(dest/pname,original);fragments.append(dict(file=pname,sha256=sha(dest/pname)))
                    prefix_keys.append({k:original[k] for k in ('history_sha256','prefix_ids_sha256','batch_size','padded_length')});prefix_keys[-1]['file']=pname
                r['fragments']=fragments;write(dest/name,r);append(dest/'closed.jsonl',dict(file=name,sha256=sha(dest/name)));episodes+=1
                if backend=='nf4':
                    expected_prefixes+=len(fragments)
                    audit_name=backend+'_'+Path(name).stem+'_audit.json';rr=read(old/'host'/audit_name)
                    rr.update(closed_file_sha256=sha(dest/name),prefix_keys=prefix_keys)
                    shutil.copyfile(old/'host'/rr['code_file'],nfhost/rr['code_file']);assert sha(nfhost/rr['code_file'])==rr['code_file_sha256']
                    write(nfhost/audit_name,rr);meta_name=Path(audit_name).stem+'_source_group.json'
                    m=read(old/'host'/meta_name);m['audit_sha256']=sha(nfhost/audit_name);write(nfhost/meta_name,m)
                    append(nfhost/'source_groups.jsonl',dict(file=meta_name,sha256=sha(nfhost/meta_name)))
            write(dest/'manifest.json',dict(MOCK_ONLY=True,backend=backend,source_manifest_sha256=oldseal if backend=='nf4' else sha(fixture/'SOURCE_MANIFEST.json')))
            write(dest/'receipt.json',dict(status='MOCK_ONLY_completed',baseline_completed=2,AB_pairs_completed=3))
            append(dest/'events.jsonl',dict(event='worker_terminal',status='MOCK_ONLY_completed'))
        write(own/'matched_inputs.json',dict(MOCK_ONLY=True,NF4_artifact_root=str(nf.resolve()),NF4_source_manifest_sha256=oldseal,
            NF4_runtime_manifest_sha256=sha(nf/'nf4/manifest.json'),NF4_condition='NAN-RAM-28',BF16_condition='NAN-BF-31'))
        write(own/'model_lane_terminal.json',dict(MOCK_ONLY=True,actual_worker_ended=True,no_model_was_started=True))
        nf_before=sha(nf/'nf4/closed.jsonl')
        with patch.object(inputs,'ROOT',fixture),patch.object(host,'verify',return_value={}),patch.object(host_budget,'RUNS',out):
            host.run(SimpleNamespace(artifact_root=str(own)))
        assert sha(nf/'nf4/closed.jsonl')==nf_before
        assert read(own/'host/receipt.json')['closed_episodes_audited']==11
        paired=read(own/'host/paired_geometry.json')
        assert paired['exact_history_and_batch_shape_prefix_pairs']==expected_prefixes
        assert paired['shared_positions_excluded_for_missing_audited_companion']==paired['shared_positions_excluded_for_history_or_shape']==0
        assert len(paired['descriptive_AB_behavior'])==6
        write(own/'host_lane_terminal.json',dict(MOCK_ONLY=True,actual_host_ended=True,exit_code=0))
        with patch.object(inputs,'ROOT',fixture),patch.object(report,'ROOT',fixture),patch.object(report,'RUNS',fixture):
            report.run(own,out_dir=out/'reports_MOCK_ONLY')
        rendered=read(out/'reports_MOCK_ONLY/nan_bf31_progress.json')
        assert sum(x['registered_episodes'] for x in rendered['groups'])==3962
        assert sum(x['closed_episodes'] for x in rendered['groups'])==22
        assert sum(x['missing_episodes'] for x in rendered['groups'])==3940
        assert not torch.cuda.is_initialized();budget.check()
        receipt=dict(MOCK_ONLY=True,status='passed',CPU_contracts=result.testsRun,synthetic_only=True,pretrained_forwards=0,CUDA_initialized=False,
            source_class='Retained synthetic MLP27 host fixture; metadata transformed only for explicit separate-root and complete-batch integration.',
            BF16_synthetic_episodes_independently_regraded=11,NF4_synthetic_audits_reused=11,NF4_raw_journal_unchanged=True,
            exact_complete_batch_geometry_prefixes=expected_prefixes,descriptive_AB_groups=6,
            registered_coverage_slots=3962,missing_slots=3940,original_frozen_sources_modified=False,sealed_test_accessed=False)
    resource=read(out/'resources.json');assert resource['status']=='passed'
    receipt['resources']=dict(samples=len(resource['samples']),maximum_sampled_private_bytes=max(x['private_bytes'] for x in resource['samples']),
        minimum_sampled_available_physical_bytes=min(x['available_physical_bytes'] for x in resource['samples']),resource_receipt_sha256=sha(out/'resources.json'))
    receipt['source_sha256']={p.name:sha(p) for p in ROOT.glob('*.py')}
    write(out/'engineering_receipt_MOCK_ONLY.json',receipt);print(__import__('json').dumps(receipt),flush=True)


if __name__=='__main__':main()
