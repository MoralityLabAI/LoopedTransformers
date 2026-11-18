"""Post-grading descriptive tables and figure; no fitting or source selection."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from experiments.nan_bpd01.analyze import memory

def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8',newline='\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    runs=Path(a.runs);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);results=read(runs/'graded/results.json');flat=read(runs/'graded/graded_cases.json');pairs=read(runs/'cohort/pairs.json')
    labels={r['case_id']:r for r in map(json.loads,(runs/'cohort/host_labels.jsonl').read_text().splitlines())};dev=[cid for cid,c in labels.items() if c['split']=='development']
    g={(r['backend'],r['case_id'],r['arm']):r for r in flat};raw={b:{cid:read(runs/b/(cid.replace(':','_')+'.json')) for cid in labels} for b in ('bf16','nf4')}
    diagnostics={};pairrows=[];transcripts=[]
    for b in ('bf16','nf4'):
        d={'memory':memory(runs/b),'pair_summary':{},'generation':{},'confusion':{},'actual_generation_paths':60}
        d['actual_generation_steps']=sum(len(raw[b][p[k]]['records'][arm]['steps']) for p in pairs for k in ('recipient','donor') for arm in ('curriculum','balanced'))+sum(len(raw[b][p['recipient']]['records'][arm]['steps']) for p in pairs for arm in ('erased','erased_repeat'))
        for arm in ('curriculum','balanced','erased','erased_repeat'):
            rr=[raw[b][cid]['records'][arm] for cid in dev];tok=[r['tokens_generated'] for r in rr]
            d['generation'][arm]={'minimum_tokens':min(tok),'maximum_tokens':max(tok),'total_tokens':sum(tok),'stop_reasons':dict(Counter(r['stop_reason'] for r in rr)),
                'generated_text_counts':dict(Counter(r['generated_text'] for r in rr))}
            d['confusion'][arm]={op:dict(Counter(g[b,cid,arm]['operator'] or 'invalid' for cid in dev if labels[cid]['operator']==op)) for op in 'INSR'}
            for cid in dev:
                r=raw[b][cid]['records'][arm];transcripts.append({'backend':b,'case_id':cid,'root':labels[cid]['root'],'target_operator':labels[cid]['operator'],
                    'arm':arm,'generated_text':r['generated_text'],'generated_ids':json.dumps(r['generated_ids']),'parser_status':r['parser']['status'],'prediction':r['parser']['operator'],
                    'tokens_generated':r['tokens_generated'],'future_correct':g[b,cid,arm]['future_correct'],'restricted_operator':g[b,cid,arm]['restricted_operator']})
        for arm in results['summaries'][b]['development']:
            rows=[]
            for pair in pairs:
                if pair['split']!='development':continue
                ids=[pair[k] for k in ('recipient','donor')];gg=[g[b,cid,arm] for cid in ids]
                row={'backend':b,'root':pair['root'],'family':pair['family'],'arm':arm,'targets':''.join(labels[cid]['operator'] for cid in ids),
                    'predictions':','.join(v['operator'] or 'invalid' for v in gg),'correct':sum(v['future_correct'] for v in gg),
                    'both_correct':int(all(v['future_correct'] for v in gg)),'restricted_both_correct':int(all(v['restricted_future_correct'] for v in gg)),
                    'different_predictions':int(gg[0]['operator']!=gg[1]['operator'])}
                pairrows.append(row);rows.append(row)
            d['pair_summary'][arm]={'roots':8,**{key:sum(r[key] for r in rows) for key in ('both_correct','restricted_both_correct','different_predictions')}}
        diagnostics[b]=d
    cross={}
    for arm in ('curriculum','balanced','erased'):
        cross[arm]={'worlds':16,'same_parsed_operator':sum(g['bf16',cid,arm]['operator']==g['nf4',cid,arm]['operator'] for cid in dev),
            'same_generated_ids':sum(raw['bf16'][cid]['records'][arm]['generated_ids']==raw['nf4'][cid]['records'][arm]['generated_ids'] for cid in dev),
            'mean_first_source_nll_NF4_minus_BF16':sum(g['nf4',cid,arm]['source_nll']-g['bf16',cid,arm]['source_nll'] for cid in dev)/16,
            'scope':'same-checkpoint paired development observation, not equivalence'}
    write(out/'diagnostics.json',{'backend':diagnostics,'cross_backend':cross,'BPD_transfer_attempted':False,'geometry_measured':False,'sealed_test_accessed':False})
    for name,rows in [('case_grades.csv',flat),('pair_grades.csv',pairrows),('generation_transcripts.csv',transcripts)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if a.plot:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
        fig,ax=plt.subplots(1,3,figsize=(13.8,4.8));colors=['#2166ac','#b35806'];arms=['balanced','curriculum','erased'];xx=np.arange(3)
        for k,b in enumerate(('bf16','nf4')):
            x=xx+(k-.5)*.32;s=results['summaries'][b]['development']
            counts=[s[arm]['future_correct'] for arm in arms];bars=ax[0].bar(x,np.array(counts)/16,.30,color=colors[k],label=b.upper())
            ax[0].bar_label(bars,labels=[str(n)+'/16' for n in counts],padding=3,fontsize=8,color=colors[k]);ax[0].plot(x,[s[arm]['restricted_future_correct']/16 for arm in arms],'o',mfc='white',mec=colors[k],ms=6)
            counts=[diagnostics[b]['pair_summary'][arm]['both_correct'] for arm in arms];bars=ax[1].bar(x,counts,.30,color=colors[k]);ax[1].bar_label(bars,labels=[str(n)+'/8' for n in counts],padding=3,fontsize=8,color=colors[k])
            ax[1].plot(x,[diagnostics[b]['pair_summary'][arm]['restricted_both_correct'] for arm in arms],'o',mfc='white',mec=colors[k],ms=6)
            counts=[s[arm]['by_operator']['R']['correct'] for arm in arms];bars=ax[2].bar(x,counts,.30,color=colors[k]);ax[2].bar_label(bars,labels=[str(n)+'/4' for n in counts],padding=3,fontsize=8,color=colors[k])
            restricted=[sum(g[b,cid,arm]['restricted_future_correct'] for cid in dev if labels[cid]['operator']=='R') for arm in arms];ax[2].plot(x,restricted,'o',mfc='white',mec=colors[k],ms=6)
        for a0 in ax:a0.set_xticks(xx,['Balanced','Reciprocal\ncurriculum','Output\nerased'])
        ax[0].set(ylim=(0,1.30),ylabel='Verified future-correct fraction',title='A  DEV worlds (16; 8 roots)');ax[0].set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']);ax[0].axhline(.75,ls='--',lw=1,color='#555555');ax[0].axhline(.25,ls=':',lw=1,color='#777777');ax[0].legend(frameon=False,ncol=2,loc='upper left')
        ax[1].set(ylim=(0,9.1),yticks=range(0,9,2),ylabel='Both targets correct (roots)',title='B  Counterfactual pair competence');ax[1].axhline(6,ls='--',lw=1,color='#555555')
        ax[2].set(ylim=(0,4.6),yticks=range(5),ylabel='Reciprocal worlds correct',title='C  Reciprocal gate (4 source roots)');ax[2].axhline(3,ls='--',lw=1,color='#555555')
        fig.suptitle('NAN-COMP-03 · bounded greedy answers · one checkpoint · development only',fontsize=14)
        fig.text(.025,.02,'Bars: bounded greedy; open circles: initial four-choice ranking. Dashed lines: curriculum gates. Dotted: constant-I prior.',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(out/'competence_summary.png',dpi=180);fig.savefig(out/'competence_summary.svg');plt.close(fig)
        svg=out/'competence_summary.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'gates':results['gates'],'memory':{b:d['memory'] for b,d in diagnostics.items()},'cross_backend':cross},indent=2))

if __name__=='__main__':main()
