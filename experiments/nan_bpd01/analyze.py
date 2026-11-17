"""Compact diagnostic tables and shareable scientific figure, after isolated grading."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def ranks(values):
    x=np.asarray(values);order=np.argsort(x,kind='stable');r=np.empty(len(x),float);start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        r[order[start:end]]=(start+end-1)/2;start=end
    return r

def memory(path):
    receipt=json.loads((path/'receipt.json').read_text());ded=shared=0;samples=0;global_used=0
    external=path/'external_memory_samples.json'
    if external.exists():
        for r in json.loads(external.read_text(encoding='utf-8-sig')):
            m=r.get('gpu_memory',[]);ded=max(ded,sum(int(x['DedicatedUsage']) for x in m));shared=max(shared,sum(int(x['SharedUsage']) for x in m));samples+=bool(m)
    else:
        for file in path.glob('memory_samples*.json'):
            for r in json.loads(file.read_text()):
                m=r.get('windows_process_gpu_memory',[])
                if isinstance(m,dict):m=[m]
                ded=max(ded,sum(int(x['DedicatedUsage']) for x in m));shared=max(shared,sum(int(x['SharedUsage']) for x in m));samples+=bool(m)
    for file in path.glob('memory_samples*.json'):
        for r in json.loads(file.read_text()):
            if r.get('driver_device_global_mib'):global_used=max(global_used,r['driver_device_global_mib'][0]*2**20)
    return {'cuda_allocated_bytes':receipt['peak_cuda_allocated_bytes'],'cuda_reserved_bytes':receipt['peak_cuda_reserved_bytes'],
        'sampled_process_dedicated_bytes':ded if samples else None,'sampled_process_shared_bytes':shared if samples else None,
        'windows_nonempty_samples':samples,'sampled_device_global_bytes':global_used,'elapsed_seconds':receipt['elapsed_seconds'],
        'shared_is_allocation_not_measured_paging':True,'head_remained_cpu':receipt['head_remained_cpu']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--graded',required=True);p.add_argument('--out',required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    runs=Path(a.runs);out=Path(a.out);graded=json.loads((Path(a.graded)/'results.json').read_text());pairs=json.loads((runs/'cohort/pairs.json').read_text())
    diagnostics={};tapes={};editrows=[];phase_rows=[];profiles=[]
    for backend in ['bf16','nf4']:
        tape={}
        for pair in pairs:
            for role in ['recipient','donor']:
                cid=pair[role];tape[cid]=json.loads((runs/backend/(cid.replace(':','_')+'.json')).read_text())
        tapes[backend]=tape
        thresholds={};phase_summary={}
        for site in ['decoder_output_pre_final_norm','loop_boundary_post_norm']:
            cal={loop:np.mean([r['phase_observations'][f'loop{loop}/layer21/{site}']['bins']['last']['channel_mean'] for r in tape.values() if r['split']=='calibration'],axis=0) for loop in [1,2]}
            thresholds[site]={'loop1_means':cal[1].tolist(),'loop2_means':cal[2].tolist(),'thresholds':((cal[1]+cal[2])/2).tolist(),'calibration_only':True}
            correct=np.zeros(5);values={loop:[] for loop in [1,2]}
            gaps=[];donordiffs=[]
            for pair in pairs:
                if pair['split']!='development':continue
                rootvalues=[]
                for role in ['recipient','donor']:
                    r=tape[pair[role]];lo=[]
                    for loop in [1,2]:
                        obs=r['phase_observations'][f'loop{loop}/layer21/{site}'];v=np.array(obs['bins']['last']['channel_mean'])
                        prediction=np.where(abs(v-cal[1])<=abs(v-cal[2]),1,2);correct+=(prediction==loop)
                        values[loop].append(v);lo.append(v)
                        phase_rows.append({'backend':backend,'root':pair['root'],'case_id':r['case_id'],'site':site,'loop':loop,'channel2279_last':v[0],
                            'channel2279_context_mean':obs['bins']['context8']['channel_mean'][0],'whole_last_token_mean_l2':obs['bins']['last']['mean_token_l2']})
                        token_values=np.array(obs['channel_mean_by_token'])
                        for bin_index,indices in enumerate(np.array_split(np.arange(len(token_values)),32)):
                            mean=token_values[indices].mean(0)
                            for channel,value in zip(obs['channels'],mean):
                                profiles.append({'backend':backend,'root':pair['root'],'case_id':r['case_id'],'site':site,'loop':loop,
                                    'profile':'normalized_position_bin32','position':bin_index,'channel':channel,'value':float(value)})
                        for offset in range(1,17):
                            profiles.append({'backend':backend,'root':pair['root'],'case_id':r['case_id'],'site':site,'loop':loop,
                                'profile':'exact_offset_from_end','position':offset,'channel':2279,'value':float(token_values[-offset,0])})
                    rootvalues.append(lo)
                    gaps.append(abs(lo[0][0]-lo[1][0]))
                donordiffs.append([abs(rootvalues[0][loop][0]-rootvalues[1][loop][0]) for loop in [0,1]])
            phase_summary[site]={'channels':[2279,2278,2280,173,1531],'dev_last_token_loop_classification_correct':correct.tolist(),
                'dev_loop_observations':16,'independent_roots':4,'loop1_mean':np.mean(values[1],axis=0).tolist(),'loop2_mean':np.mean(values[2],axis=0).tolist(),
                'channel2279_mean_phase_gap':float(np.mean(gaps)),'channel2279_mean_donor_recipient_absolute_gap_by_loop':np.mean(donordiffs,axis=0).tolist()}
        for cid,r in tape.items():
            if r['split']!='development':continue
            for name,e in r['interventions'].items():
                if 'dose' not in e:continue
                dose=e['dose'];current=e['current'];delayed=e['delayed']
                editrows.append({'backend':backend,'root':r['root'],'case_id':cid,'arm':name,**dose,
                    'current_full_vocab_max_lp_error':current['full_vocab_max_logprob_error_vs_reference'],
                    'current_full_vocab_lp_l2':current['full_vocab_logprob_l2_vs_reference'],
                    'delayed_full_vocab_lp_l2':delayed['full_vocab_logprob_l2_vs_reference'],
                    'delayed_full_vocab_max_lp_error':delayed['full_vocab_max_logprob_error_vs_reference'],
                    'next_state_l2':e['next_state_l2'],'cosine_to_donor_next_displacement':e['cosine_to_donor_next_displacement'],
                    'delayed_lp_l2_per_physical_l2':delayed['full_vocab_logprob_l2_vs_reference']/dose['actual_l2'] if dose['actual_l2'] else None})
        selected=[r for r in editrows if r['backend']==backend]
        dark=[r for r in selected if r['scored_tail_identical']]
        diagnostics[backend]={'memory':memory(runs/backend),'phase_calibration':thresholds,'phase':phase_summary,
            'physical_dose':{'edits_measured':len(selected),'ineligible':sum(not r['eligible'] for r in selected),'zero_dose':sum(r['zero_dose'] for r in selected),
                'maximum_relative_error':max(r['relative_error'] for r in selected),'current_dark_edits':len(dark),
                'current_dark_max_full_vocab_error':max(r['current_full_vocab_max_lp_error'] for r in dark),
                'whole_state_norm_ratio_range':[min(r['whole_state_norm_ratio'] for r in selected),max(r['whole_state_norm_ratio'] for r in selected)]}}
    alignment=[]
    for cid,b in tapes['bf16'].items():
        if b['split']!='development':continue
        n=tapes['nf4'][cid]
        for arm in ['native1','native2','open_loop','equal_compute']:
            x=b['records'][arm]['scores'];y=n['records'][arm]['scores'];rho=float(np.corrcoef(ranks(x),ranks(y))[0,1])
            alignment.append({'case_id':cid,'root':b['root'],'arm':arm,'spearman64':rho})
    write(out/'diagnostics.json',{'backend':diagnostics,'cross_backend_rank_alignment':alignment,
        'scope':'rank alignment and phase diagnostics; learned BPD subspace alignment is gated and unrun','sealed_test_accessed':False})
    write(out/'intervention_geometry.json',editrows);write(out/'phase_by_world.json',phase_rows)
    for name,rows in [('intervention_geometry.csv',editrows),('phase_by_world.csv',phase_rows),('rank_alignment.csv',alignment),('phase_position_profiles.csv',profiles)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if a.plot:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors={'bf16':'#285b9e','nf4':'#d06b24'};fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for ax,site,title in [(axes[0,0],'decoder_output_pre_final_norm','Raw layer 21 channel 2279'),(axes[0,1],'loop_boundary_post_norm','After loop RMSNorm')]:
            for backend in ['bf16','nf4']:
                subset=[r for r in phase_rows if r['backend']==backend and r['site']==site]
                for cid in sorted({r['case_id'] for r in subset}):
                    v=[r['channel2279_last'] for r in subset if r['case_id']==cid];ax.plot([1,2],v,'o-',color=colors[backend],alpha=.5,ms=3)
                ax.plot([],[],color=colors[backend],label=backend.upper())
            ax.set(title=title,xlabel='Native loop',ylabel='Last-token activation',xticks=[1,2]);ax.legend(frameon=False)
        names=['native1','native2','calibrated','open_loop','equal_compute','frequency'];labels=['One','Two','Temp','Open','Equal','Freq']
        ax=axes[0,2]
        for j,backend in enumerate(['bf16','nf4']):
            vals=[graded['backend_results'][backend]['task'][n]['future12_correct']/8 for n in names]
            ax.bar(np.arange(6)+(j-.5)*.36,vals,.36,color=colors[backend],label=backend.upper())
        ax.set(title='Verified future correct at budget 12',ylabel='Solve fraction',xticks=np.arange(6),xticklabels=labels,ylim=(0,1));ax.legend(frameon=False)
        ax=axes[1,0];names=['phase_context_suppress','phase_context_replace','random_dense_dark_741','random_dense_dark_947'];labs=['Suppress','Replace','Random A','Random B']
        for j,backend in enumerate(['bf16','nf4']):
            for i,name in enumerate(names):
                vals=[r['delayed_lp_l2_per_physical_l2'] for r in editrows if r['backend']==backend and r['arm']==name]
                ax.scatter([i+(j-.5)*.2]*len(vals),vals,color=colors[backend],marker='o' if j==0 else 'x')
        ax.set(title='Delayed response of current-dark edits',ylabel='Full-vocabulary LP L2 / physical L2',xticks=range(4),xticklabels=labs,yscale='log')
        ax=axes[1,1];keys=['cuda_allocated_bytes','cuda_reserved_bytes','sampled_process_dedicated_bytes','sampled_process_shared_bytes']
        for j,backend in enumerate(['bf16','nf4']):ax.bar(np.arange(4)+(j-.5)*.36,[diagnostics[backend]['memory'][k]/2**30 for k in keys],.36,color=colors[backend])
        ax.set(title='Separate memory measures',ylabel='GiB',xticks=range(4),xticklabels=['Allocated','Reserved','Dedicated*','Shared*']);ax.text(.02,.95,'* sampled Windows process allocation',transform=ax.transAxes,va='top',fontsize=8)
        ax=axes[1,2]
        for i,name in enumerate(['native1','native2','open_loop','equal_compute']):
            vals=[r['spearman64'] for r in alignment if r['arm']==name];ax.scatter([i]*len(vals),vals,color='#546a7b',s=18,alpha=.7)
        ax.set(title='BF16 versus NF4 program rankings',ylabel='Spearman rho across 64 scores',xticks=range(4),xticklabels=['One','Two','Open','Equal'],ylim=(-1,1));ax.axhline(0,color='gray',lw=.7)
        fig.suptitle('NAN-BPD-01 development pilot | 4 source roots, 8 worlds, one checkpoint',fontsize=14)
        fig.savefig(out/'nan_bpd01_summary.png',dpi=180);fig.savefig(out/'nan_bpd01_summary.pdf');plt.close(fig)
    print(json.dumps({'backend':diagnostics,'alignment_mean_by_arm':{n:float(np.mean([r['spearman64'] for r in alignment if r['arm']==n])) for n in ['native1','native2','open_loop','equal_compute']}},indent=2))

if __name__=='__main__':main()
