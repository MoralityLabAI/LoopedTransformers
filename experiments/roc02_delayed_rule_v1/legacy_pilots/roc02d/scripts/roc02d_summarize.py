"""Standard-library aggregator for ROC-02D registered calibration-only input controls."""
from __future__ import annotations
import csv
import hashlib
import io
import gzip
import json
import math
import statistics as st
import zipfile
from pathlib import Path

SEEDS=(2026111601,2026111603,2026111607,2026111609,2026111613,2026111619)
DEPTHS=(8,16)
INDICES=(2,3,4,5,6,7)
ARMS=('full_legal_flip','input_soft_radius','input_induced_midpoint_radius',
      'hidden_observed_radius','hidden_dark_radius','hidden_isotropic_radius')
OBS=('task_only','joint')
RADIUS=0.2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(folder):
    files=[];cases=[];arm_rows=[]
    for depth in DEPTHS:
        for seed in SEEDS:
            path=folder/f'roc02d_R{depth}_S{seed}.json'
            receipt=json.loads(path.read_text())
            if (receipt['seed'],receipt['depth'])!=(seed,depth) or not receipt['calibration_only'] or receipt['sealed_test_opened']:
                raise AssertionError('source identity or calibration-only flag invalid')
            if len(receipt['rows'])!=12:
                raise AssertionError('incomplete six-case times two-adapter study')
            files.append({'seed':seed,'depth':depth,'file':path.name,'sha256':sha(path),
                          'checkpoint_sha256':receipt['frozen_model_checkpoint_sha256'],
                          'adapter_sha256':receipt['frozen_adapter_sha256']})
            for r in receipt['rows']:
                if r['case_index'] not in INDICES or r['observer_arm'] not in OBS or r['label_original']==r['label_flipped']:
                    raise AssertionError('unexpected causal input label/observer')
                if r['soft_t'] is None or not (0<=r['soft_t']<=1):
                    raise AssertionError('registered matched input control was unavailable')
                cases.append(r)
                if len(r['effects'])!=len(ARMS):raise AssertionError('arm cardinality')
                for e in r['effects']:
                    if e['arm'] not in ARMS or not e['available']:
                        raise AssertionError('invalid declared arm')
                    if e['arm'] not in ('full_legal_flip',):
                        if abs(e['midpoint_shift_norm']-RADIUS)>2e-5:
                            raise AssertionError('matched physical L2 radius violated')
                    if e['original_logits_change_norm']<0 or e['observer_mid_readout_change_norm']<0:
                        raise AssertionError('negative output change norm')
                    arm_rows.append({
                        'seed':seed,'depth':depth,'case_index':r['case_index'],
                        'observer_arm':r['observer_arm'],'arm':e['arm'],
                        'source_family':r['source_family'],'flipped_family':r['flipped_family'],
                        'label_original':r['label_original'],'label_flipped':r['label_flipped'],
                        'soft_t':r['soft_t'],'full_discrete_flip_midpoint_L2':r['full_discrete_flip_midpoint_L2'],
                        'midpoint_shift_norm':e['midpoint_shift_norm'],
                        'original_logits_change_norm':e['original_logits_change_norm'],
                        'observer_mid_readout_change_norm':e['observer_mid_readout_change_norm'],
                        'future_logits_tangent_residual_norm':e['future_logits_tangent_residual_norm'],
                        'true_original_margin_change':e['true_original_margin_change'],
                        'original_label_probability_direction':e['original_label_probability_direction'],
                        'argmax_changed':e['argmax_changed']})
    assert len(cases)==144 and len(arm_rows)==864 and len(files)==12
    assert len({(r['depth'],r['seed'],r['case_index'],r['observer_arm'],r['arm']) for r in arm_rows})==864
    return files,cases,arm_rows


def summary(folder):
    receipts,cases,rows=load(folder)
    paired=[]
    by_depth={}
    for depth in DEPTHS:
        for seed in SEEDS:
            subset=[r for r in rows if r['depth']==depth and r['seed']==seed]
            out={'depth':depth,'seed':seed,'n_cases':6,'calibration_only':True}
            for observer in OBS:
                for arm in ARMS:
                    items=[r for r in subset if r['observer_arm']==observer and r['arm']==arm]
                    assert len(items)==6
                    for field in ('observer_mid_readout_change_norm','original_logits_change_norm'):
                        out[f'{observer}_{arm}_{field}']=st.mean(float(r[field]) for r in items)
            paired.append(out)
        d=[r for r in paired if r['depth']==depth]
        flat=[r for r in rows if r['depth']==depth]
        change={}
        for arm in ARMS:
            key='observer_mid_readout_change_norm'
            task=[r[f'task_only_{arm}_{key}'] for r in d]
            joint=[r[f'joint_{arm}_{key}'] for r in d]
            delta=[b-a for a,b in zip(task,joint)]
            change[arm]={'task_only_mean':st.mean(task),'joint_mean':st.mean(joint),
                         'joint_minus_task_mean':st.mean(delta),
                         'joint_increases_seed_count':sum(x>0 for x in delta),
                         'joint_decreases_seed_count':sum(x<0 for x in delta)}
        by_depth[str(depth)]={'model_seeds':list(SEEDS),'n_source_cases':36,
                              'n_per_case_effect_rows':len(flat),'armwise_observer_mid_change':change,
                              'mean_full_discrete_flip_midpoint_L2':st.mean(r['full_discrete_flip_midpoint_L2'] for r in cases if r['depth']==depth and r['observer_arm']=='task_only'),
                              'mean_soft_input_interpolation_t':st.mean(r['soft_t'] for r in cases if r['depth']==depth and r['observer_arm']=='task_only')}
    report={'study':'ROC-02D-calibration-only-input-versus-hidden-perturbation-v1',
            'registration_commit':'714f149d6dabff24a3ef4bcec566ba7d93f4cf0d',
            'model_cells':12,'unique_case_model_pairs':72,
            'observer_case_rows':len(cases),'arm_effect_rows':len(rows),'calibration_only':True,
            'sealed_test_opened':False,'source_exact_label_changes':72,
            'soft_input_radius_success':72,
            'input_control_semantics':'full legal source edit changes exact answer; softened input is outside the discrete source grammar',
            'note':'Full legal edits are not L2=0.2 cost-matched. Model/adapter weights and previously selected gamma unchanged. No prospective type-identification claim.',
            'by_depth':by_depth,'source_receipts':receipts}
    return report,paired,rows


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def publish(input_dir:Path,output_dir:Path):
    output_dir.mkdir(parents=True,exist_ok=True)
    report,paired,rows=summary(input_dir)
    write_csv(output_dir/'roc02d_paired_by_seed.csv',paired)
    raw=output_dir/'roc02d_per_case_effects.csv'
    write_csv(raw,rows)
    data=raw.read_bytes()
    compressed=gzip.compress(data,mtime=0)
    (output_dir/'roc02d_per_case_effects.csv.gz').write_bytes(compressed)
    raw.unlink()
    (output_dir/'roc02d_summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    archive=output_dir/'roc02d_raw_receipts.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for r in report['source_receipts']:
            p=input_dir/r['file'];z.write(p,arcname='receipts/'+p.name)
    manifest={p.name:sha(p) for p in output_dir.glob('*') if p.is_file()}
    (output_dir/'roc02d_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return report


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();r=publish(a.input_dir,a.output_dir)
    print(json.dumps({k:r[k] for k in ('model_cells','unique_case_model_pairs','observer_case_rows','arm_effect_rows','by_depth')},indent=2,sort_keys=True))

if __name__=='__main__':main()
