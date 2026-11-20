import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
out=Path(__file__).parent
import json
record=json.loads((out/'routing_root_values.json').read_text())
rows=record['unit_rows']
roots=np.array([row['root'] for row in rows])
fixed=np.array([row['fixed'] for row in rows])/record['n_cases_per_root']*100
online=np.array([row['online'] for row in rows])/record['n_cases_per_root']*100
teacher=np.array([row['teacher_scheduled'] for row in rows])/record['n_cases_per_root']*100
dense=np.array([row['dense'] for row in rows])/record['n_cases_per_root']*100
fig,ax=plt.subplots(figsize=(8.0,3.45))
y=np.arange(len(roots))[::-1]
for i in range(len(roots)):
    ax.plot([fixed[i],online[i]],[y[i],y[i]],color='#a5b5be',lw=1.4,zorder=1)
ax.scatter(fixed,y,color='#748995',s=29,marker='s',label='Fixed sparse route',zorder=2)
ax.scatter(online,y,color='#1D6671',s=41,marker='o',label='Online sparse route',zorder=3)
ax.scatter(teacher,y,color='#AA7027',s=38,marker='^',label='Teacher-informed route',zorder=3)
ax.scatter(dense,y,color='#24364D',s=42,marker='|',linewidths=2.5,label='Dense teacher',zorder=3)
ax.set_yticks(y,labels=[str(z) for z in roots]);ax.tick_params(axis='y',labelsize=8.5,length=0,pad=8)
ax.set_xlabel('Exact Sudoku4 board accuracy at depth 8 (%)',fontsize=9,labelpad=8)
ax.set_xlim(-2,104);ax.set_xticks([0,20,40,60,80,100]);ax.tick_params(axis='x',labelsize=8.4)
ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
for spine in ax.spines.values():spine.set_visible(False)
ax.legend(loc='upper center',bbox_to_anchor=(.50,-.24),frameon=False,ncol=2,fontsize=8.2,handletextpad=.5,columnspacing=1.5)
fig.tight_layout(pad=1.2)
fig.savefig(out/'routing_by_root.pdf',bbox_inches='tight',pad_inches=.12)
fig.savefig(out/'routing_by_root.png',dpi=200,bbox_inches='tight',pad_inches=.12)
print('figure',out/'routing_by_root.pdf')
