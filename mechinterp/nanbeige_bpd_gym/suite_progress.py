"""Aggregate-only progress publisher for the frozen wider-suite pilot."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from experiments.nan_gym05.common import read,write,sha
from experiments.nan_gym05.grade import grade

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_gym05');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    runs=Path(a.runs);result,_private=grade(runs,allow_partial=True);del _private
    result['updated_UTC']=datetime.now(timezone.utc).isoformat()
    result['source_manifest_sha256']=sha('experiments/nan_gym05/SOURCE_MANIFEST.json')
    result['publication_policy']='Aggregates and sanitized coverage only. No local routing text, IDs, raw predictions or vectors.'
    result['pointer_readout_screen']={}
    for b,rr in result['summaries'].items():
        hop=next((r for r in rr if r['split']=='development' and r['suite']=='pointer' and r['condition']=='hop1'),None)
        copy=next((r for r in rr if r['split']=='development' and r['suite']=='pointer' and r['condition']=='copy'),None)
        complete=bool(hop and copy and hop['records']==copy['records']==16)
        result['pointer_readout_screen'][b]='pending' if not complete else ('pass' if hop['correct']>=12 and copy['correct']==16 else 'fail')
    for b,v in result['coverage'].items():
        v.pop('root_ids',None)
        if v['status']=='not_loaded' and (runs/b).exists():v['status']='loading'
        elif v['status']=='running' and not (runs/b/'linear_baselines/manifest.json').exists():v['status']='extracting_weight_baselines'
    result['tower_diagnostics']={}
    for b in ('bf16','nf4'):
        probe=runs/'probes'/(b+'_aggregate_probes.json');audit=runs/'audit'/(b+'_independent.json')
        if not probe.exists() or not audit.exists():continue
        checked=read(audit)
        if checked['status']!='passed':raise ValueError('Do not publish unverified tower results')
        layers=[]
        for layer in read(probe)['layers']:
            number=layer['layer_zero_based'];fit=runs/b/'sae_tower'/f'layer{number}_receipt.json'
            layers.append({'layer_zero_based':number,'dictionary_sha256':layer['dictionary_sha256'],
                'fit_manifest_sha256':layer['fit_manifest_sha256'],'fit_receipt':read(fit),
                'physical_vs_normalized_pre_max_abs':layer['physical_vs_normalized_pre_max_abs'],
                'groups':[{k:v for k,v in g.items() if k!='feature_active_fraction_root_macro'} for g in layer['groups']],
                'loop_gaps':layer['loop_gaps']})
        result['tower_diagnostics'][b]={'independent_audit':{k:v for k,v in checked.items() if k!='artifact_sha256'},
            'audit_file_sha256':sha(audit),'probe_file_sha256':sha(probe),
            'layers':layers,'scope':'CAL-only fitted dictionaries; observational diagnostics, no causal transfer'}
    out=Path(a.out);write(out/'nan_gym05_progress.json',result)
    lines=['# NAN-GYM-05 streaming development report','',f"Updated {result['updated_UTC']}",'',
        'Frozen source-only registration; one checkpoint, BF16/NF4 distinct conditions. '+('Partial source-order counts are not final accuracy estimates.' if result['partial'] else 'Both native backends are complete; results remain descriptive exposed-development evidence.'),'',
        '| Backend | Status | Root groups complete |','|---|---|---:|']
    for b,v in result['coverage'].items():lines.append(f"| {b.upper()} | {v['status']} | {v['roots']}/89 |")
    lines+=['','New pointer readout screen: '+', '.join(b.upper()+' '+v for b,v in result['pointer_readout_screen'].items())+'. This only nominates a task for future causal design; original algebraic gates stay failed and no BPD transfer is enabled.']
    lines+=['','| Backend | Split | Suite | Condition | Correct | Format valid | Root groups |','|---|---|---|---|---:|---:|---:|']
    for b,rr in result['summaries'].items():
        for r in rr:lines.append(f"| {b.upper()} | {r['split']} | {r['suite']} | {r['condition']} | {r['correct']}/{r['records']} | {r['format_valid']}/{r['records']} | {r['roots']} |")
    if result['tower_diagnostics']:
        lines+=['','Completed SAE towers use 256 CAL vectors per layer, shared across both loops. Each fitted and random initial dictionary is checked in physical residual coordinates. These are observational reconstruction diagnostics; normalized MSE is not task accuracy. Full suite/loop group statistics and independent audit counts are in the companion JSON.','',
            '| Backend | Layer (zero based) | CAL normalized MSE | CAL features ever active / 128 |',
            '|---|---:|---:|---:|']
        for b,data in result['tower_diagnostics'].items():
            for layer in data['layers']:
                fit=layer['fit_receipt']
                lines.append(f"| {b.upper()} | {layer['layer_zero_based']} | {fit['normalized_CAL_MSE']:.6f} | {fit['features_ever_active_on_CAL']}/128 |")
        lines+=['','Sparse-code indices are dictionary specific: BF16 and NF4 feature numbers cannot be directly equated. SVD/Haar direction geometry is a weight baseline; genuine trained VPD/BPD remains pending.']
    lines+=['','Routing is category classification, not Intellect/GSM8K/ARC QA solution accuracy. ARC-1/2 are local public grid microbenches. Sudoku primary is the upstream exact reference; alternate valid solutions are reported separately in JSON. Storyworld correctness executes the complete plan under the native rival.',
        '','SAE fitting uses only the 32 CAL pointer roots. Weight SVD is an algebraic baseline, not a trained VPD/BPD result. No model edits or BPD transfer run; the original 168-case test stays sealed.','']
    (out/'NAN_GYM05_PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({b:{k:v[k] for k in ('roots','status')} for b,v in result['coverage'].items()})

if __name__=='__main__':main()
