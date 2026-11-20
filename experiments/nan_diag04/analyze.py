"""Descriptive component tables, transcripts and standalone scientific figure."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from experiments.nan_bpd01.analyze import memory

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    runs=Path(a.runs);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r=read(runs/'graded/results.json');flat=read(runs/'graded/graded_records.json');metrics=read(runs/'graded/root_metrics.json')
    labels={c['root']:c for c in map(json.loads,(runs/'cohort/host_labels.jsonl').read_text().splitlines())};dev=[k for k,c in labels.items() if c['split']=='development'];diag={};raw={}
    for b in ('bf16','nf4'):
        raw[b]={k:read(runs/b/f'root_{k}.json') for k in labels};d={'memory':memory(runs/b),'format_by_component':{},'text_counts':{},'actual_paths':240,'actual_token_steps':0}
        for tape in raw[b].values():d['actual_token_steps']+=sum(len(x['steps']) for x in tape['records'].values())
        for arm in ('free','scaffold','copy','select','table','coded','labelcopy','codecopy','erased','erased_repeat'):
            rr=[v for v in flat if v['backend']==b and v['split']=='development' and v['arm'].split('/')[0]==arm]
            d['format_by_component'][arm]={'records':len(rr),'valid':sum(x['format_valid'] for x in rr),'tokens':sum(x['tokens_generated'] for x in rr)}
            d['text_counts'][arm]=dict(Counter(x['text'] for x in rr))
        diag[b]=d
    cross={}
    for arm in raw['bf16'][dev[0]]['records']:
        cross[arm]={'roots':8,'same_value':sum(raw['bf16'][k]['records'][arm]['parser']['value']==raw['nf4'][k]['records'][arm]['parser']['value'] for k in dev),
            'same_ids':sum(raw['bf16'][k]['records'][arm]['generated_ids']==raw['nf4'][k]['records'][arm]['generated_ids'] for k in dev)}
    (out/'diagnostics.json').write_text(json.dumps({'backend':diag,'cross_backend':cross,'evidence_class':'descriptive, one checkpoint','BPD_transfer_attempted':False},indent=2)+'\n',encoding='utf-8',newline='\n')
    with (out/'graded_transcripts.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    rootrows=[{'backend':b,'root':k,'split':labels[int(k)]['split'],**v} for b,mm in metrics.items() for k,v in mm.items()]
    with (out/'root_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rootrows[0]));w.writeheader();w.writerows(rootrows)
    if a.plot:
        import numpy as np
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
        fig,ax=plt.subplots(1,3,figsize=(14,4.8));colors=['#2166ac','#b35806'];groups=[['free_full','scaffold_full','copy_full'],['select_R','table_R','coded_R'],['select_alt','table_alt','coded_alt']]
        names=[['Free\ncoefficients','Recurrence\nscaffold','Numeric\ncopy'],['Raw\nselection','Operator\nreference','Neutral-code\nreference'],['Raw\nselection','Operator\nreference','Neutral-code\nreference']]
        for j,keys in enumerate(groups):
            for i,b in enumerate(('bf16','nf4')):
                counts=[r['summaries'][b]['development'][key] for key in keys];bars=ax[j].bar(np.arange(3)+(i-.5)*.32,counts,.30,color=colors[i],label=b.upper())
                ax[j].bar_label(bars,labels=[str(int(v))+'/8' for v in counts],padding=3,fontsize=9,color=colors[i])
            ax[j].set(xticks=range(3),xticklabels=names[j],ylim=(0,10),yticks=range(0,9,2));ax[j].axhline(6,ls='--',color='#555555',lw=1)
        ax[0].set(title='A  Full reciprocal coefficients',ylabel='Source roots correct (8 DEV roots)');ax[0].legend(frameon=False,ncol=2,loc='upper left')
        ax[1].set_title('B  Reciprocal-world label selection');ax[2].set_title('C  Alternative-world label selection')
        fig.suptitle('NAN-DIAG-04 · computation versus matching · one checkpoint · development only',fontsize=14)
        fig.text(.025,.02,'Full computation requires all four coefficients. Dashed: 6/8 descriptive high-performance threshold, not confirmation. Copy controls require 8/8.',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(out/'component_summary.png',dpi=180);fig.savefig(out/'component_summary.svg');plt.close(fig)
        svg=out/'component_summary.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'patterns':r['diagnostic_pattern'],'memory':{b:d['memory'] for b,d in diag.items()}},indent=2))

if __name__=='__main__':main()
