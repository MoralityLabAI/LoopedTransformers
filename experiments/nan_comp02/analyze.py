"""Descriptive competence diagnostics and scientific figure; no fitting."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import numpy as np
from experiments.nan_bpd01.analyze import memory

def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    runs=Path(a.runs);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    results=read(runs/'graded/results.json');flat=read(runs/'graded/graded_cases.json');pairs=read(runs/'cohort/pairs.json')
    labels={r['case_id']:r for r in map(json.loads,(runs/'cohort/host_labels.jsonl').read_text().splitlines())}
    raw={b:{cid:read(runs/b/(cid.replace(':','_')+'.json')) for cid in labels} for b in ('bf16','nf4')}
    dev=[cid for cid,c in labels.items() if c['split']=='development'];grades={(r['backend'],r['case_id'],r['arm']):r for r in flat}
    diagnostics={};cross={};pairrows=[]
    arms=['direct_native1','direct_native2','worked_native1','worked_native2','erased_native2']
    for b in ('bf16','nf4'):
        d={'memory':memory(runs/b),'confusion':{},'pair_source_response':{},'calibration_overlap_sensitivity':{}}
        for arm in arms:
            table={op:dict.fromkeys(['I','N','S','R','invalid'],0) for op in 'INSR'}
            for cid in dev:table[labels[cid]['operator']][grades[b,cid,arm]['free_operator'] or 'invalid']+=1
            d['confusion'][arm]=table
        d['worked2_first_tokens']=dict(Counter(str(raw[b][cid]['records']['worked_native2']['free_top_token_id'])+':'+raw[b][cid]['records']['worked_native2']['free_top_token_text'] for cid in dev))
        for arm in arms:
            free_different=restricted_different=0;rows=[]
            for pair in pairs:
                if pair['split']!='development':continue
                ids=[pair[k] for k in ('recipient','donor')];gg=[grades[b,cid,arm] for cid in ids]
                free_different+=int(gg[0]['free_operator']!=gg[1]['free_operator']);restricted_different+=int(gg[0]['constrained_operator']!=gg[1]['constrained_operator'])
                row={'backend':b,'root':pair['root'],'family':pair['family'],'arm':arm,'target_operators':''.join(labels[cid]['operator'] for cid in ids),
                     'free_predictions':','.join(g['free_operator'] or 'invalid' for g in gg),'free_correct':sum(g['future_correct'] for g in gg),'both_correct':int(all(g['future_correct'] for g in gg)),
                     'restricted_correct':sum(g['constrained_future_correct'] for g in gg)}
                pairrows.append(row);rows.append(row)
            d['pair_source_response'][arm]={'roots':8,'different_free_predictions':free_different,'different_restricted_predictions':restricted_different,
                'both_correct':sum(r['both_correct'] for r in rows)}
        sensitivity=[cid for cid in dev if labels[cid]['root']!=182705]
        d['calibration_overlap_sensitivity']={'excluded_root':182705,'cases':len(sensitivity),'roots':7,
            'worked2_free_correct':sum(grades[b,cid,'worked_native2']['future_correct'] for cid in sensitivity),
            'worked2_restricted_correct':sum(grades[b,cid,'worked_native2']['constrained_future_correct'] for cid in sensitivity),
            'scope':'post-audit descriptive sensitivity only; all 16 worlds retained in frozen gate'}
        diagnostics[b]=d
    for arm in arms:
        free_same=restricted_same=0;tvs=[];nllchanges=[]
        for cid in dev:
            bb=raw['bf16'][cid]['records'][arm];nn=raw['nf4'][cid]['records'][arm]
            free_same+=int(bb['free_top_token_id']==nn['free_top_token_id']);restricted_same+=int(bb['constrained_operator']==nn['constrained_operator'])
            prob=lambda r: np.exp(np.array([r['class_logprobs'][op] for op in 'INSR'])-max(r['class_logprobs'].values()))
            pb,pn=prob(bb),prob(nn);pb/=pb.sum();pn/=pn.sum();tvs.append(float(np.abs(pb-pn).sum()/2))
            nllchanges.append(grades['nf4',cid,arm]['source_nll']-grades['bf16',cid,arm]['source_nll'])
        cross[arm]={'cases':16,'same_free_token':free_same,'same_restricted_operator':restricted_same,'mean_conditional_four_operator_total_variation':float(np.mean(tvs)),
            'mean_source_nll_NF4_minus_BF16':float(np.mean(nllchanges)),'scope':'observed same-checkpoint agreement; not quantization equivalence'}
    write(out/'diagnostics.json',{'backend':diagnostics,'cross_backend':cross,'geometry_measured':False,'BPD_transfer_attempted':False,'sealed_168_case_test_accessed':False})
    with (out/'case_grades.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    with (out/'pair_grades.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(pairrows[0]));w.writeheader();w.writerows(pairrows)
    if a.plot:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
        fig,ax=plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1.5,1,1.3]});colors=['#2166ac','#b35806'];xx=np.arange(4)
        clean=['direct_native1','direct_native2','worked_native1','worked_native2']
        for k,b in enumerate(('bf16','nf4')):
            s=results['summaries'][b]['development'];x=xx+(k-.5)*.32
            bars=ax[0].bar(x,[s[arm]['future_correct']/16 for arm in clean],.30,color=colors[k],label=b.upper());ax[0].bar_label(bars,labels=[str(s[arm]['future_correct'])+'/16' for arm in clean],padding=3,fontsize=8,color=colors[k])
            ax[0].plot(x,[s[arm]['constrained_future_correct']/16 for arm in clean],'o',mfc='white',mec=colors[k],ms=6)
        ax[0].axhline(.75,ls='--',lw=1,color='#555555');ax[0].set(xticks=xx,xticklabels=['Direct\nloop 1','Direct\nloop 2','Worked\nloop 1','Worked\nloop 2'],ylim=(0,1.07),ylabel='Verified future-correct fraction',title='A  Fresh DEV worlds (16; 8 roots)')
        fig.text(.03,.02,'Bars: unrestricted greedy; open circles: four-choice ranking. Dashed lines: frozen main-arm gates.',fontsize=9);ax[0].set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']);ax[0].legend(loc='upper left',ncol=2,frameon=False)
        for k,b in enumerate(('bf16','nf4')):
            s=results['gates'][b];bars=ax[1].bar(np.arange(2)+(k-.5)*.32,[s['both_correct_pairs'],s['erased_both_correct_pairs']],.30,color=colors[k]);ax[1].bar_label(bars,labels=[str(s['both_correct_pairs'])+'/8',str(s['erased_both_correct_pairs'])+'/8'],padding=4,fontsize=8,color=colors[k])
        ax[1].axhline(6,ls='--',lw=1,color='#555555');ax[1].set(xticks=[0,1],xticklabels=['Worked\nloop 2','Output\nerased'],ylim=(0,8.7),yticks=range(0,9,2),ylabel='Both distinct targets correct (roots)',title='B  Pair competence gate')
        keys=['cuda_allocated_bytes','cuda_reserved_bytes','sampled_process_dedicated_bytes','sampled_process_shared_bytes']
        for k,b in enumerate(('bf16','nf4')):
            m=diagnostics[b]['memory'];ax[2].bar(np.arange(4)+(k-.5)*.32,[(m[key] or 0)/2**30 for key in keys],.30,color=colors[k])
        ax[2].set(xticks=range(4),xticklabels=['CUDA\nallocated','CUDA\nreserved','WDDM\ndedicated','WDDM\nshared'],ylabel='GiB (separate counters)',title='C  Observed peak memory')
        ax[2].set_ylim(0,max(diagnostics[b]['memory']['sampled_process_dedicated_bytes']/2**30 for b in ('bf16','nf4'))*1.25)
        ax[2].text(.02,.98,'WDDM sampled; shared ≠ measured paging',transform=ax[2].transAxes,fontsize=8,va='top')
        fig.suptitle('NAN-COMP-02 · new unary source task · one checkpoint · development only',fontsize=14)
        fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(out/'competence_summary.png',dpi=180);fig.savefig(out/'competence_summary.svg');svg=out/'competence_summary.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n',encoding='utf-8',newline='\n');plt.close(fig)
    print(json.dumps({'cross_backend':cross,'memory':{b:d['memory'] for b,d in diagnostics.items()}},indent=2))
if __name__=='__main__':main()
