"""Regenerate manuscript schematics and comparison plots from unchanged CSVs."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'
BACK=ROOT/'revision/editorial_structure_before'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})

def save(fig,name):
    for ext in ['pdf','png']:
        old=FIG/f'{name}.{ext}'
        if old.exists() and not (BACK/old.name).exists():shutil.copy2(old,BACK/old.name)
        fig.savefig(old,dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def workflow():
    fig,ax=plt.subplots(figsize=(10,5.3))
    ax.set(xlim=(0,10),ylim=(0,5.3));ax.axis('off')
    stages=[
      (1.6,4.45,'Dataset audit\nCanonical undirected pairs\nMethods 2.1'),
      (5,4.45,'Split construction\nProtein novelty and similarity\nMethods 2.2'),
      (8.4,4.45,'Representations and models\nFrozen embeddings and descriptors\nMethods 2.3'),
      (8.4,2.65,'Negative-set comparisons\nTopology, matching, and substitution\nMethods 2.4'),
      (5,2.65,'Metrics and calibration\nRanking and probability quality\nMethods 2.5'),
      (1.6,2.65,'Graph-aware inference\nComponent and seed sensitivity\nMethods 2.6'),
      (1.6,.85,'Second benchmark\nPooled and within-species controls\nMethods 2.7'),
      (5,.85,'Results and interpretation\nEstimates, uncertainty, and scope\nResults 3.1–3.10'),
    ]
    for x,y,t in stages:
        ax.add_patch(FancyBboxPatch((x-1.5,y-.52),3,1.04,boxstyle='round,pad=.05',facecolor='#eef3f8',edgecolor='#36546f'))
        ax.text(x,y,t,ha='center',va='center',fontsize=9)
    for a,b in zip(stages,stages[1:]):
        x,y,_=a;xx,yy,_=b
        if y==yy:
            sign=np.sign(xx-x);start=(x+sign*1.56,y);end=(xx-sign*1.56,yy)
        else:start=(x,y-.59);end=(xx,yy+.59)
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'-|>','color':'#36546f','lw':1.3})
    ax.set_title('Study workflow',fontsize=14,pad=12)
    save(fig,'figure1_study_workflow')

def split_design():
    fig,axes=plt.subplots(1,5,figsize=(12,3.8))
    titles=['R0: random pairs','R1: seen–seen','R2: one unseen','R3: both unseen','R3: cluster-controlled']
    notes=['Protein reuse is allowed;\nnovelty is not constrained.','Both test endpoints occur\nin training; the pair is new.','Exactly one test endpoint\nis absent from training.','Both test endpoints are\nabsent from training.','Both test endpoints and\ntheir assigned clusters\nare absent from training.']
    states=[['seen','new'],['seen','seen'],['seen','new'],['new','new'],['new','new']]
    for i,(ax,title,note,status) in enumerate(zip(axes,titles,notes,states)):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,fontsize=10)
        ax.text(.5,.9,'Example test pair',ha='center',fontsize=9,color='#485769')
        ax.plot([.27,.73],[.57,.57],color='#667788',lw=2,zorder=1)
        for x,st in zip([.27,.73],status):
            color='#237a58' if st=='seen' else '#d77924'
            if i==0:color='#8d99a5'
            if i==4:
                ax.add_patch(FancyBboxPatch((x-.18,.36),.36,.42,boxstyle='round,pad=.02',facecolor='#fff5e9',edgecolor='#d77924',linestyle='--'))
            ax.scatter([x],[.57],s=700,color=color,zorder=3)
            ax.text(x,.57,'A' if x<.5 else 'B',ha='center',va='center',color='white',weight='bold')
        ax.text(.5,.13,note,ha='center',va='center',fontsize=8.5)
    fig.text(.5,.02,'Green: observed in training   •   Orange: absent from training   •   Gray: unconstrained   •   Dashed outline: assigned sequence cluster',ha='center',fontsize=8.5)
    fig.suptitle('Evaluation regimes (Methods 2.2)',fontsize=13)
    fig.tight_layout(rect=(0,.06,1,.90),w_pad=1)
    save(fig,'figure3_split_design')

def comparisons(source,group_col,name,title):
    df=pd.read_csv(ROOT/source)
    regimes=['R0','R1','R2','R3','R3-50','R3-40','R3-30','R3-20']
    assert set(df.Regime)==set(regimes)
    xpos=np.array([0,1.8,2.8,3.8,5.6,6.6,7.6,8.6])
    fig,axes=plt.subplots(1,3,figsize=(12.5,4))
    colors=['#4c78a8','#e28b25','#219878','#b3538f']
    groups=list(df[group_col].unique())
    for ax,metric in zip(axes,['ROC-AUC','PR-AUC','MCC']):
        for j,group in enumerate(groups):
            g=df[df[group_col]==group].set_index('Regime').loc[regimes]
            offset=(j-(len(groups)-1)/2)*.055
            values=g[metric].to_numpy()
            ax.scatter(xpos+offset,values,s=29,color=colors[j],label=group,zorder=3)
            # Connect only the matched novelty strata, leaving independent designs separate.
            ax.plot(xpos[1:4]+offset,values[1:4],color=colors[j],lw=1.2)
        if metric!='PR-AUC':ax.axhline(.5 if metric=='ROC-AUC' else 0,color='#777777',ls='--',lw=.8)
        ax.axvline(.9,color='#cccccc',lw=.7);ax.axvline(4.7,color='#cccccc',lw=.7)
        ax.set_xticks(xpos,regimes,rotation=45,ha='right');ax.set_title(metric)
        ax.set_xlabel('Evaluation regime');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Score')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=len(groups),frameon=False,fontsize=9,bbox_to_anchor=(.5,-.01))
    fig.suptitle(title,fontsize=13)
    fig.tight_layout(rect=(0,.08,1,.93))
    save(fig,name)

if __name__=='__main__':
    workflow();split_design()
    comparisons('tables/tableS23_figure3_data.csv','Model','figure4_generalization_degradation','AAC+CTD model performance across evaluation regimes')
    comparisons('tables/tableS24_figure4_data.csv','Representation','figure5_representation_comparison','Representation performance with the same MLP classifier')
    print('Refreshed Figures 1, 2, 4, and 5 from procedures and unchanged source data.')
