"""Assemble complete primary and complete secondary comparisons; never select cases."""
import argparse,json,shutil
from pathlib import Path
import ram03_core as c
from evaluate import summarize

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('primary','secondary','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((a.primary/'evaluation_report.json').read_text())
    later=json.loads((a.secondary/'evaluation_report.json').read_text())
    assert later['status']=='complete'
    for key in ('config_sha256','dataset_sha256','selected_models','source_hashes','original_oracles'):
        assert prior[key]==later[key],key
    trajectories={}
    for experiment,root,policies in [('primary',a.primary,('native','frequency','loop32','loop64','flat64')),('secondary',a.secondary,('native','loop64'))]:
        for suite,count in [('prospective',12),('posthoc_fc13',16)]:
            for model in policies:
                key='__'.join((experiment,suite,model));path=root/(key+'.jsonl');rows=c.read_rows(path)
                assert len(rows)==count and len({(r['source_file'],r['theorem']) for r in rows})==count
                assert all((r['experiment'],r['suite'],r['model'])==(experiment,suite,model) for r in rows)
                trajectories[key]=rows;shutil.copyfile(path,a.out/path.name)
                assert c.digest(path)==c.digest(a.out/path.name)
    assert sum(map(len,trajectories.values()))==196
    # The interrupted process wrote the full primary receipt before beginning secondary.
    assert len([r for r in prior['summaries'] if r['experiment']=='primary'])==10
    resources=[prior.get('resources'),later['resources']]
    # An interrupt can precede the final receipt; recover peaks from its immutable samples.
    if resources[0] is None:
        samples=c.read_rows(a.primary/'resources.jsonl')
        resources[0]=dict(peaks={k:max(r.get(k,0) for r in samples) for k in ('tree_rss_bytes','cuda_allocated_bytes','cuda_reserved_bytes')},
          minimum_available_ram_bytes=min(r['available_ram_bytes'] for r in samples),samples=len(samples),abort=None,
          monitor_period_seconds=.1,nvidia_smi_period_seconds=2.)
        resources[0]['peaks']['dedicated_used_bytes']=max((r.get('dedicated_vram') or {}).get('used_bytes',0) for r in samples)
        assert not any(r.get('abort') for r in samples)
    merged=dict(peaks={k:max(r['peaks'].get(k,0) for r in resources) for k in set().union(*(r['peaks'] for r in resources))},
       minimum_available_ram_bytes=min(r['minimum_available_ram_bytes'] for r in resources),samples=sum(r['samples'] for r in resources),
       abort=None,monitor_period_seconds=.1,nvidia_smi_period_seconds=2.)
    assert not any(r['abort'] for r in resources)
    report={k:later[k] for k in ('experiment','config_sha256','dataset_sha256','selected_models','source_hashes','original_oracles')}
    excluded=[r for path in sorted(a.primary.glob('secondary__*__*.jsonl')) for r in c.read_rows(path)]
    report.update(status='complete',summaries=summarize(trajectories),resources=merged,
      preserved_initial_secondary=dict(attempts=len(excluded),startup_timeouts=sum(r['reason']=='startup_or_request_timeout' for r in excluded),
        path=str(a.primary),included_in_final_coverage=False),
      component_runs=[dict(budget='primary',directory=str(a.primary),report_sha256=c.digest(a.primary/'evaluation_report.json')),
                      dict(budget='secondary',directory=str(a.secondary),report_sha256=c.digest(a.secondary/'evaluation_report.json'))],
      operational_retry='Full secondary comparison repeated after unexplained initial-state startup timeout episode. All old attempts retained; no timeout or policy enlarged; no individual case selected.')
    c.write_json(a.out/'evaluation_report.json',report)
    print('RAM03_ASSEMBLED_COMPLETE',sum(map(len,trajectories.values())),flush=True)
if __name__=='__main__':main()
