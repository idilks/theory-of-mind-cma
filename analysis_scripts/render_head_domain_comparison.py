"""Render saved strict-belief-selection comparisons; performs no inference."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/modularsymbtom-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    out=root/'results/aggregated/head_domain_comparison'
    c=pd.read_csv(out/'cma_head_comparisons.csv')
    members=pd.read_csv(root/'results/aggregated/head_evidence_inventory/head_membership.csv')
    names=members[members.membership.eq('belief')].head_name.tolist()
    fig,axes=plt.subplots(1,2,figsize=(11,7),sharey=True)
    for ax,key,title in zip(axes,['C1_minus_C2','C1_minus_K1'],['Belief − photo','Belief − code']):
        rows=c[c.name.eq(key+':equal_cell_mean')].set_index('head_name').loc[names]
        value=rows.mean_difference.to_numpy()
        error=np.array([value-rows.simultaneous_lower.to_numpy(),rows.simultaneous_upper.to_numpy()-value])
        ax.errorbar(value,np.arange(len(names)),xerr=error,fmt='o',color='#24577a',ecolor='#66889e',capsize=3)
        ax.axvline(0,color='0.45',lw=1,ls='--')
        ax.set_title(title);ax.set_xlabel('Difference in CMA logit-margin change')
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(names)),names);axes[0].invert_yaxis()
    fig.suptitle('The 12 strictly belief-only selected heads',fontsize=15,y=.98)
    fig.text(.5,.91,'Equal-weight observed template/direction means; positive = larger belief intervention effect',ha='center',fontsize=10)
    fig.text(.5,.02,'Approximate simultaneous 95% bootstrap bounds across heads and planned contrasts.\nCondition-specific success filtering remains; these are not independent replication results.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.89));fig.savefig(out/'cma_belief_only_contrasts.png',dpi=180);plt.close(fig)
    print(out/'cma_belief_only_contrasts.png')


if __name__=='__main__':main()
