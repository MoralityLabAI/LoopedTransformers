"""Standalone matched IO18 score/dose/resource plot from one immutable audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

COMMIT='f419a7a1a75cb16d4cb6c9748a2927c249244dcd'
SOURCE='reports/nanbeige_bpd_gym/nan_fit16_io18_progress.json'
SOURCE_SHA='67129eba2f7d995638e9a43d35fd1cbcd3eab0a00e6b93274147be18affe321c'
ROUTES=('all_one','component_only','static_CAL_mean','clean_informed','reached_state')


def load():
    from .io18_results import validate
    raw=subprocess.check_output(['git','show',COMMIT+':'+SOURCE]);assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    data=json.loads(raw);validate(data);values={};records=[]
    for backend in ('bf16','nf4'):
        cell=next(c for c in data['backends'][backend]['cells'] if c['passes']==2)
        assert cell['status']=='passed' and cell['training']['optimizer_steps']==16
        groups={g['mode']:g for g in cell['final_groups'] if g['split']=='development' and g['loop']==2}
        assert set(groups)==set(ROUTES)
        values[backend]=dict(correct=[],KL=[],local_write=[],resources=[])
        for route in ROUTES:
            g=groups[route];assert g['contexts']==32 and g['source_roots']==16
            for panel,field in (('correct','correct'),('KL','full_vocabulary_KL'),('local_write','local_intervention_relative_l2_at_actual_reached_input')):
                v=g[field] if panel=='correct' else g[field]['equal_root_mean']
                values[backend][panel].append(v)
                records.append(dict(panel=panel,backend=backend,route=route,passes=2,split='development',loop=2,
                    source_field=field,value=v,worlds=32,independent_source_roots=16))
        resource=[('CUDA cumulative allocated',cell['CUDA_peak_allocated_bytes']),('CUDA cumulative reserved',cell['CUDA_peak_reserved_bytes']),
            ('Added saved CPU activation peak',cell['saved_tensor_statistics']['peak_added_cpu_bytes'])]
        for label,count in resource:
            v=count/2**30;values[backend]['resources'].append(v)
            records.append(dict(panel='resources',backend=backend,label=label,bytes=count,value_GiB=v,passes=2))
    assert len(records)==36
    return values,records


def main():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    from .evidence_index import below_normal
    below_normal();values,records=load()
    producer='mechinterp/nanbeige_bpd_gym/plot_io18_comparison.py';head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    code=subprocess.check_output(['git','show',head+':'+producer]);assert code.replace(b'\r\n',b'\n')==Path(producer).read_bytes().replace(b'\r\n',b'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    figure,axes=plt.subplots(2,2,figsize=(12,8.4));colors={'bf16':'#27699C','nf4':'#D87529'}
    labels=['Native','Component\nonly','Static CAL\nmean mask','Clean\ninformed','Reached\nstate'];x=np.arange(5);width=.36
    specs=[('correct','A  DEV loop2 correctness','Correct worlds / 32'),('KL','B  Full-vocabulary readout change','KL(clean || changed), nats'),
        ('local_write','C  Local physical edit at reached input','Module-write relative L2')]
    for axis,(key,title,ylabel) in zip(axes.flat,specs):
        for backend,offset in (('bf16',-width/2),('nf4',width/2)):
            bars=axis.bar(x+offset,values[backend][key],width,color=colors[backend],label=backend.upper(),zorder=3)
            fmt='%d' if key=='correct' else '%.3f'
            axis.bar_label(bars,fmt=fmt,padding=3,fontsize=8)
        axis.set_title(title,loc='left',fontweight='bold',fontsize=11);axis.set_xticks(x,labels);axis.set_ylabel(ylabel)
        axis.grid(axis='y',alpha=.2,zorder=0);axis.margins(y=.24)
        if key=='correct':axis.set_ylim(0,37);axis.set_yticks([0,8,16,24,32])
    axis=axes.flat[3];xr=np.arange(3)
    for backend,offset in (('bf16',-width/2),('nf4',width/2)):
        bars=axis.bar(xr+offset,values[backend]['resources'],width,color=colors[backend],label=backend.upper(),zorder=3)
        axis.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    axis.set_title('D  Resources through T2',loc='left',fontweight='bold',fontsize=11)
    axis.set_xticks(xr,['CUDA\nallocated','CUDA\nreserved','Added saved\nCPU activations']);axis.set_ylabel('GiB');axis.grid(axis='y',alpha=.2,zorder=0);axis.margins(y=.22)
    handles,legend=axes.flat[0].get_legend_handles_labels();figure.legend(handles,legend,loc='upper right',bbox_to_anchor=(.975,.965),ncol=2,frameon=False)
    figure.suptitle('Matched BF16 / NF4 trained routes: IO18 development pilot',x=.055,y=.97,ha='left',fontsize=15,fontweight='bold')
    figure.text(.055,.925,'T2 fits: 16 CAL-only updates each. DEV: 16 reused roots, two worlds per root. All five routes retained.',fontsize=10)
    figure.text(.055,.045,'KL and local-edit panels show equal-root means; doses are unmatched. Static controls limit adaptive specificity.\nCUDA peaks are cumulative within backend; saved-activation CPU peaks are per cell, not total RAM. No causal or equivalence claim.',fontsize=9,color='#444444',va='bottom')
    figure.subplots_adjust(left=.075,right=.97,bottom=.17,top=.855,wspace=.3,hspace=.48)
    out=Path('reports/nanbeige_bpd_gym/figures');out.mkdir(exist_ok=True)
    paths=[]
    for extension in ('png','svg'):
        path=out/('nan_io18_matched_routes.'+extension);figure.savefig(path,dpi=180);paths.append(path)
    plt.close(figure)
    receipt=dict(source_commit=COMMIT,source_file=SOURCE,source_sha256=SOURCE_SHA,producer_commit=head,
        producer_sha256=hashlib.sha256(code).hexdigest(),plotted_values=records,
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},new_model_calls=0,new_grades=0,sealed_test_accessed=False)
    (out/'nan_io18_matched_routes_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(dict(source_commit=COMMIT,plotted_values=len(records),artifacts=[str(p) for p in paths]))


if __name__=='__main__':main()
