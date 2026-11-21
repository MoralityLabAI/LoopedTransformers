"""Readable, complete-stratum presentation of committed strict SAE/factor audits.

No raw-run access, fitting, model call, grade, selected feature or formal test.
Ranges and mean absolute weight cosines below are post hoc display summaries.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

COMMIT='fefa31be596ac37a0bdf5e2000e8a5efd79c23f4'
SOURCE='reports/nanbeige_bpd_gym/nan_xref25_xref19_progress.json'
FIT_COMMIT='f419a7a1a75cb16d4cb6c9748a2927c249244dcd'
SEAL='7d9ac0ff9d809900fc967283c3ec8b24fd4fefa9da410d54ec4227ab2c3a1975'
DATASETS={'gym05':644,'comp14':480,'ctx08':582}
ARRAYS=('encoder_vpd_cosine','decoder_vpd_cosine','encoder_vpd_preactivation_sensitivity')


def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path])
def digest(value):return hashlib.sha256(value).hexdigest()


def flatten(matrix):
    assert len(matrix)==128 and all(len(row)==8 for row in matrix)
    values=[v for row in matrix for v in row]
    assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
    return values


def scalar_row(prefix,group):
    omitted=[];result=dict(prefix)
    for key,value in group.items():
        if isinstance(value,(list,dict)):omitted.append(key)
        else:
            assert not isinstance(value,float) or math.isfinite(value)
            assert key not in result;result[key]=value
    result['array_fields_retained_in_source_JSON']=';'.join(sorted(omitted))
    return result


def load():
    raw=blob(COMMIT,SOURCE);data=json.loads(raw)
    assert data['experiment']=='NAN-XREF-25' and data['source_manifest_sha256']==SEAL
    assert data['new_model_calls']==data['new_grades']==data['formal_inferential_comparisons']==0
    assert data['sealed_test_accessed'] is False and data['original_failures_preserved'] is True
    fitpath='reports/nanbeige_bpd_gym/nan_fit16_io18_progress.json';fitraw=blob(FIT_COMMIT,fitpath);fit=json.loads(fitraw)
    for backend in ('bf16','nf4'):
        value=data['backends'][backend];assert value['status']=='passed'
        audit=value['repair_audit'];assert audit['source_manifest_sha256']==SEAL and audit['new_model_calls']==0
        assert audit['original_failed_attempt_unchanged'] and audit['mechanical_expression_changes']['physical_geometry_float64_direction_casts']==1
        checked={c['passes']:c for c in audit['original_arithmetic_audit']['cells']}
        assert {c['passes'] for c in value['cells']}==set(checked)=={1,2}
        for cell in value['cells']:
            depth=cell['passes'];check=checked[depth]
            assert cell['status']=='completed' and check['status']=='passed' and check['cached_vectors']==1706
            assert check['sparse_code_values_checked']==1706*4*128 and check['physical_geometry_cells']==18
            original=next(c for c in fit['backends'][backend]['cells'] if c['passes']==depth)
            assert original['status']=='passed' and original['training']['optimizer_steps']==16
            assert cell['checkpoint_sha256']==original['checkpoint_sha256']
            assert {d['dataset']:d['vectors'] for d in cell['datasets']}==DATASETS
            assert len(cell['geometry'])==18
            for dataset in cell['datasets']:
                assert {t['layer_zero_based'] for t in dataset['towers']}=={5,11,17,21}
                for tower in dataset['towers']:
                    assert sum(g['vectors'] for g in tower['groups'])==dataset['vectors']
                assert sum(g['vectors'] for g in dataset['learned_groups'])==dataset['vectors']
            assert sum(128*8*len(d['learned_groups']) for d in cell['datasets'])==check['root_correlation_cells_checked']
    return data,dict(source_commit=COMMIT,source_file=SOURCE,source_sha256=digest(raw),
        fit_source_commit=FIT_COMMIT,fit_source_file=fitpath,fit_source_sha256=digest(fitraw))


def build(data):
    scalars=[];tower_rows=[];factor_rows=[];geometry=[];coverage=[];weight_rows=[]
    for backend in ('bf16','nf4'):
        for cell in sorted(data['backends'][backend]['cells'],key=lambda c:c['passes']):
            depth=cell['passes'];prefix=dict(backend=backend,fitted_depth=depth)
            for dataset in cell['datasets']:
                name=dataset['dataset'];base=prefix|dict(dataset=name)
                for tower in dataset['towers']:
                    layer=tower['layer_zero_based'];groups=tower['groups']
                    def span(field):
                        values=[g[field] for g in groups];return [min(values),max(values)]
                    tower_rows.append(base|dict(layer_zero_based=layer,registered_strata=len(groups),vectors=dataset['vectors'],
                        normalized_MSE_range=span('normalized_MSE_root_mean'),random_initial_MSE_range=span('random_initial_MSE_root_mean'),
                        MSE_improvement_range=span('MSE_improvement_root_mean')))
                    for kind,bank in (('SAE',groups),('paired_loop_gap',tower['loop_gaps'])):
                        for group in bank:scalars.append(scalar_row(base|dict(record_kind=kind,layer_zero_based=layer),group))
                groups=dataset['learned_groups'];undefined=0
                for group in groups:
                    matrix=group['feature_vs_centered_routed_energy_root_correlation']
                    assert len(matrix)==128 and all(len(v)==8 for v in matrix)
                    missing=sum(v is None for row in matrix for v in row)
                    assert missing==group['undefined_correlation_cells'];undefined+=missing
                    if group['source_root_groups']<3:assert missing==1024
                    scalars.append(scalar_row(base|dict(record_kind='cached_component',layer_zero_based=21),group))
                def span(field):
                    values=[g[field] for g in groups];return [min(values),max(values)]
                factor_rows.append(base|dict(registered_strata=len(groups),
                    learned_relative_L2_range=span('learned_pure_component_relative_L2'),initial_SVD_relative_L2_range=span('initial_SVD_relative_L2'),
                    matched_norm_Haar_relative_L2_range=span('matched_norm_Haar_relative_L2'),
                    defined_correlation_entries=len(groups)*1024-undefined,total_correlation_entries=len(groups)*1024,
                    out_of_fit_depth_strata=sum(g['out_of_fit_depth'] for g in groups)))
            bases={g['basis'] for g in cell['geometry']};assert len(bases)==3
            for basis in sorted(bases):
                cases=[g for g in cell['geometry'] if g['basis']==basis];assert len(cases)==6
                for case in cases:
                    assert all(case[name]==cases[0][name] for name in ARRAYS)
                    assert case['zero_norm_counts']==cases[0]['zero_norm_counts']
                identifier=f'{backend}:T{depth}:{basis}';case=cases[0]
                arrays={name:flatten(case[name]) for name in ARRAYS}
                matrix_sha=digest(json.dumps({name:case[name] for name in ARRAYS},sort_keys=True,separators=(',',':')).encode())
                geometry.append(prefix|dict(basis=basis,unique_geometry_id=identifier,numerical_matrix_sha256=matrix_sha,registered_repeated_cases=6,
                    mean_absolute_encoder_cosine=sum(abs(v) for v in arrays[ARRAYS[0]])/1024,
                    mean_absolute_decoder_cosine=sum(abs(v) for v in arrays[ARRAYS[1]])/1024,zero_norm_counts=case['zero_norm_counts']))
                for g in cases:
                    coverage.append(prefix|dict(unique_geometry_id=identifier,basis=basis,dataset=g['dataset'],
                        application_loop=g['application_loop'],out_of_fit_depth=g['out_of_fit_depth']))
                for feature in range(128):
                    for component in range(8):
                        weight_rows.append(dict(unique_geometry_id=identifier,feature_index=feature,component_index=component,
                            encoder_cosine=case[ARRAYS[0]][feature][component],decoder_cosine=case[ARRAYS[1]][feature][component],
                            encoder_preactivation_sensitivity=case[ARRAYS[2]][feature][component]))
    assert len(tower_rows)==48 and len(factor_rows)==12 and len(geometry)==12 and len(coverage)==72 and len(weight_rows)==12288
    numerical={}
    for group in geometry:numerical.setdefault(group['numerical_matrix_sha256'],[]).append(group['unique_geometry_id'])
    numerical_groups=[dict(matrix_sha256=h,labelled_groups=ids) for h,ids in sorted(numerical.items())]
    return dict(towers=tower_rows,factors=factor_rows,geometry=geometry,numerical_geometry_groups=numerical_groups,scalar_rows=scalars,
        geometry_coverage_rows=coverage,unique_weight_geometry_rows=weight_rows)


def csv_text(rows):
    fields=sorted(set().union(*(row.keys() for row in rows)));stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return stream.getvalue()


def span(values):return f'{values[0]:.5g} to {values[1]:.5g}'


def render(result,provenance):
    negative=sum(r['MSE_improvement_range'][0]<0 for r in result['towers'])
    defined=sum(r['defined_correlation_entries'] for r in result['factors']);total=sum(r['total_correlation_entries'] for r in result['factors'])
    lines=['# Audited CAL-only SAE towers and learned component cross-references','',
        'Both NF4 and BF16 passed the separate strict XREF25 independent audit at fitted depths1 and2. Each cell rechecks all1706 cached vectors,873472 sparse-code values,129024 correlation entries and18 physical geometry cases. Original XREF19/XREF24 failures remain failed; the decoder orientation and geometry dtype are repaired only in the separately registered XREF25 condition. No tolerance was widened, worker/model replayed, or grade added.','',
        f'Some registered strata reconstruct worse with the trained SAE than its random-initial control: negative improvement appears in{negative} of the48 backend/depth/dataset/layer range rows. Those rows repeat dictionaries across depths. Across all cached-factor strata,{defined} of{total} registered correlation entries are defined; the rest remain null under the original root-count/constant-coordinate rules.','',
        'The same eight fixed CAL-only SAE dictionaries (four layers per backend,128 features/top4) are reused across fitted depths and datasets. Learned components/controllers had16 real CAL-only optimizer steps at each depth. Cached applications are observational: component reconstruction and centered routed energy do not equal the actual sphere-rescaled online write or establish task mediation.','',
        'Unit:reused source root/family on one checkpoint. GYM05 has89 root groups/644 phase vectors, COMP14 has24 roots/480 vectors, CTX08 five families/582 vectors. These counts repeat across fits and layers; they are not pooled independent population replications.','',
        '## SAE reconstruction across every registered stratum','',
        'Each range below is the minimum to maximum of the EXISTING equal-root group means across all registered CAL/DEV, loop and task/control strata in that row. It is not an overall root mean, uncertainty interval or new hypothesis test. Strata remain separate in the scalar CSV; all128-feature activity vectors and128x8 correlations remain in the full source JSON. Negative improvements are retained.','',
        '| Backend | Fit depth | Dataset | Layer | Strata | Phase vectors | Trained normalized MSE range | Random-initial MSE range | Improvement range |',
        '|---|---:|---|---:|---:|---:|---|---|---|']
    for r in result['towers']:
        lines.append(f"| {r['backend']} | {r['fitted_depth']} | {r['dataset']} | {r['layer_zero_based']} | {r['registered_strata']} | {r['vectors']} | {span(r['normalized_MSE_range'])} | {span(r['random_initial_MSE_range'])} | {span(r['MSE_improvement_range'])} |")
    lines+=['','## Cached component reconstruction and undefined correlations','',
        'Ranges again retain all existing CAL/DEV/loop/control strata. Error is pure component prediction versus native physical module output at the cached clean input. Haar factors have matched input-column/output-row norms. Correlation counts are matrix entries, not independent tests: fewer than3 roots or constant coordinates stay null. T1 applications at loop2 are outside fit depth.','',
        '| Backend | Fit depth | Dataset | Strata | Learned relative L2 range | Initial SVD range | Matched-norm Haar range | Defined / all correlation entries | Out-of-depth strata |',
        '|---|---:|---|---:|---|---|---|---:|---:|']
    for r in result['factors']:
        lines.append(f"| {r['backend']} | {r['fitted_depth']} | {r['dataset']} | {r['registered_strata']} | {span(r['learned_relative_L2_range'])} | {span(r['initial_SVD_relative_L2_range'])} | {span(r['matched_norm_Haar_relative_L2_range'])} | {r['defined_correlation_entries']} / {r['total_correlation_entries']} | {r['out_of_fit_depth_strata']} |")
    lines+=['','## Fixed physical weight geometry','',
        f"Only the layer21 down_proj output-to-residual identity-coordinate bridge is used. The same weight matrices repeat across three datasets/two application loops; all72 registered cases map to12 labelled backend/depth/basis groups. Their numerical arrays have{len(result['numerical_geometry_groups'])} distinct fingerprints: initial SVD controls also repeat across fitted depths. Every matrix entry was checked equal before grouping repetitions. Mean absolute cosines are explicitly post hoc display summaries over ALL128x8 entries, without selected features/components, significance tests or causal interpretation. The legacy unique-geometry CSV name refers to unique labelled coefficient keys, not independent numerical bases. Signs and every sensitivity remain in that CSV. Independently fitted backend SAE indices are not subtracted; the finite diagonal/permutation gauge check does not identify arbitrary rotated component directions.",'',
        '| Backend | Fit depth | Basis | Repeated registered cases | Mean absolute encoder cosine | Mean absolute decoder cosine | Zero encoder / decoder / output rows |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in result['geometry']:
        z=r['zero_norm_counts']
        lines.append(f"| {r['backend']} | {r['fitted_depth']} | {r['basis']} | {r['registered_repeated_cases']} | {r['mean_absolute_encoder_cosine']:.6g} | {r['mean_absolute_decoder_cosine']:.6g} | {z['encoder_rows']} / {z['decoder_rows']} / {z['VPD_vectors']} |")
    lines+=['','## Provenance and limits','',
        f"Immutable source: `{provenance['source_commit']}`, SHA-256 `{provenance['source_sha256']}`. Fit checkpoint/16-step qualification is cross-bound to `{provenance['fit_source_commit']}`.",'',
        'All outcomes and prior repair failures were exposed before this presentation. Physical neural dose:none in this cached analysis. Scientific causal floor, matched-sham floor and causal power remain uncalibrated; planned/actual formal comparisons0, verdict REDESIGN_OR_PILOT_ONLY. More untouched roots, a prospectively calibrated causal contrast and an independent sealed readout would be needed for confirmation. No original failed algebraic/FIT07 gate is changed, and no donor transfer or equivalence is claimed. The168-case test stays sealed; the continuous goal remains active.','',
        f"[Full immutable audit JSON](https://github.com/MoralityLabAI/LoopedTransformers/blob/{COMMIT}/{SOURCE}) · [All scalar strata](nan_xref25_scalar_strata.csv) · [All unique weight entries](nan_xref25_unique_weight_geometry.csv) · [All72 geometry-case bindings](nan_xref25_geometry_coverage.csv) · [Presentation receipt](nan_xref25_results_receipt.json) · [Separate repair](NAN_XREF25_XREF19_PROGRESS.md) · [Online score/dose comparison](NAN_FIT16_IO18_RESULTS.md)",'']
    return '\n'.join(lines)


def main():
    from .evidence_index import below_normal
    below_normal();data,provenance=load();result=build(data)
    producer='mechinterp/nanbeige_bpd_gym/xref25_results.py';head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    code=blob(head,producer);assert code.replace(b'\r\n',b'\n')==Path(producer).read_bytes().replace(b'\r\n',b'\n')
    out=Path('reports/nanbeige_bpd_gym');files={
        'NAN_XREF25_RESULTS.md':render(result,provenance),'nan_xref25_scalar_strata.csv':csv_text(result['scalar_rows']),
        'nan_xref25_unique_weight_geometry.csv':csv_text(result['unique_weight_geometry_rows']),
        'nan_xref25_geometry_coverage.csv':csv_text(result['geometry_coverage_rows'])}
    for name,text in files.items():(out/name).write_text(text,encoding='utf-8',newline='\n')
    receipt=provenance|dict(producer_commit=head,producer_sha256=digest(code),
        row_counts={k:len(v) for k,v in result.items()},
        artifact_sha256={name:digest((out/name).read_bytes()) for name in files},
        display_summary_exposure='Post hoc ranges/all-entry means over already public audited arrays; no selected feature or new formal test.',
        new_model_calls=0,new_grades=0,new_optimizer_steps=0,formal_inferential_comparisons=0,sealed_test_accessed=False)
    (out/'nan_xref25_results_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(dict(source_commit=COMMIT,row_counts=receipt['row_counts'],new_model_calls=0))


if __name__=='__main__':main()
