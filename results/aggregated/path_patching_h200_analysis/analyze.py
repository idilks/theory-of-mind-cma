"""All 41 heads: full repaired grid, cellwise inference and fixed-grid contrasts.

No model inference. Run audit_repairs.py first. Across-cell uncertainty does not
assume independence or pair prompts by row number. Domain bounds are averages of
simultaneous cell bounds; contrast bounds subtract endpoints. This is deliberately
conservative. Bounds are approximate bootstrap intervals with ten prompts/cell.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

P=Path(__file__).parent
ROOT=P.resolve().parents[2]
DOMAINS=('C1','C2','K1')
KINDS=('target','target_minus_control')
SEED=20261023
DRAWS=50000
ALPHA=.05

def simultaneous_cell(y, rng, draws=DRAWS, families=14):
    """Enumerated max-|mean| sign test and joint centered bootstrap bounds."""
    n,m=y.shape
    assert n==10 and np.isfinite(y).all()
    mean=y.mean(0)
    signs=2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1
    maxima=np.max(abs(signs@y/n),axis=1)
    p=np.minimum(1.,families*np.mean(maxima[:,None]>=abs(mean)[None,:]-1e-12,axis=0))
    se=y.std(0,ddof=1)/np.sqrt(n);valid=se>0
    maxima=[]
    if valid.any():
        for start in range(0,draws,1000):
            weights=rng.multinomial(n,np.full(n,1/n),size=min(1000,draws-start))/n
            maxima.extend(np.max(abs((weights@y[:,valid]-mean[valid])/se[valid]),axis=1))
        critical=float(np.quantile(maxima,1-ALPHA/families))
    else: critical=0.
    lo=np.where(valid,mean-critical*se,np.nan);hi=np.where(valid,mean+critical*se,np.nan)
    return mean,lo,hi,p,critical,int((~valid).sum())


def mean_bounds(tables):
    # numpy.mean propagates unestimated bounds; pandas.mean would silently drop them.
    return {k:np.mean(np.stack([t[k].to_numpy() for t in tables]),axis=0) for k in ('mean','lower','upper')}


def main():
    audit=json.loads((P/'audit.json').read_text());assert audit['all_14_cells_passed']
    meta=pd.read_csv(P/'connections.csv',keep_default_na=False);assert len(meta)==2218 and meta.sender.nunique()==41
    z=np.load(P/'validated_effects.npz'); cells=pd.read_csv(P/'control_audit_by_cell.csv')
    assert cells.passed.all() and cells.passed.dtype==bool
    rng=np.random.default_rng(SEED);frames=[];settings=[];cellmeans=[]
    for cid, domain in cells[['cell','domain']].itertuples(index=False,name=None):
        x=z[cid+'__target'];c=z[cid+'__control'];y=np.concatenate([x,x-c],1)
        mean,lo,hi,p,critical,nzero=simultaneous_cell(y,rng)
        for j,kind in enumerate(KINDS):
            sl=slice(j*len(meta),(j+1)*len(meta))
            f=meta.copy();f['cell']=cid;f['domain']=domain;f['kind']=kind;f['n_prompts']=10
            f['mean']=mean[sl];f['lower']=lo[sl];f['upper']=hi[sl];f['exact_maxstat_p_14_cells']=p[sl]
            f['detected']=p[sl]<=ALPHA;f['positive_prompts']=(y[:,sl]>0).sum(0);f['negative_prompts']=(y[:,sl]<0).sum(0)
            f['matched_control_mean']=c.mean(0);frames.append(f)
        cm=meta.copy();cm['cell']=cid;cm['domain']=domain;cm['target_mean']=x.mean(0);cm['control_mean']=c.mean(0);cm['target_minus_control_mean']=(x-c).mean(0);cellmeans.append(cm)
        settings.append(dict(cell=cid,bootstrap_critical=critical,zero_variance_columns=nzero))
        print(cid,'corrected detections (target, target-control)',int((p[:2218]<=ALPHA).sum()),int((p[2218:]<=ALPHA).sum()),flush=True)
    est=pd.concat(frames,ignore_index=True);est.to_csv(P/'cell_estimates.csv',index=False)
    cm=pd.concat(cellmeans,ignore_index=True);cm.to_csv(P/'cell_means.csv',index=False)
    est[est.detected].to_csv(P/'detected_cell_connections.csv',index=False)
    # Named fixed-grid averages: all cells, each direction, and story-template-matched
    # comparisons against the code template. No new significance family is introduced:
    # all intervals derive from the same simultaneous cell bounds.
    grids={'all_templates_both_directions':{d:cells.loc[cells.domain==d,'cell'].tolist() for d in DOMAINS}}
    for direction in ('ABA','ABB'):
        grids[direction]={d:[c for c in grids['all_templates_both_directions'][d] if c.endswith('_'+direction)] for d in DOMAINS}
    for template in ('food_truck','hair_styling','library_book'):
        grids[template+'_both_directions']={d:[c for c in grids['all_templates_both_directions'][d] if template in c or d=='K1'] for d in DOMAINS}
    grouped={(cid,k):g.reset_index(drop=True) for (cid,k),g in est.groupby(['cell','kind'],sort=False)}
    domainframes=[]
    for grid,groups in grids.items():
        for d,cids in groups.items():
            for k in KINDS:
                f=meta.copy();f['grid']=grid;f['domain']=d;f['kind']=k;f['cells']=';'.join(cids);f['n_cells']=len(cids);f['n_prompt_measurements']=10*len(cids)
                for stat,arr in mean_bounds([grouped[cid,k] for cid in cids]).items():f[stat]=arr
                f['positive_supported']=f.lower>0;f['negative_supported']=f.upper<0
                f['matched_control_mean']=np.mean([z[cid+'__control'].mean(0) for cid in cids],axis=0)
                domainframes.append(f)
    domains=pd.concat(domainframes,ignore_index=True);domains.to_csv(P/'domain_estimates.csv',index=False)
    df={(grid,d,k):g.reset_index(drop=True) for (grid,d,k),g in domains.groupby(['grid','domain','kind'],sort=False)}
    contrasts=[]
    for grid in grids:
        for k in KINDS:
            for left,right in [('C1','C2'),('C1','K1'),('C2','K1')]:
                a=df[grid,left,k];b=df[grid,right,k];f=meta.copy();f['grid']=grid;f['kind']=k;f['left']=left;f['right']=right
                f['difference']=a['mean']-b['mean'];f['lower']=a.lower-b.upper;f['upper']=a.upper-b.lower
                f['excludes_zero']=(f.lower>0)|(f.upper<0);contrasts.append(f)
    contrasts=pd.concat(contrasts,ignore_index=True);contrasts.to_csv(P/'domain_differences.csv',index=False)
    contrasts[contrasts.excludes_zero].to_csv(P/'supported_domain_differences.csv',index=False)
    allgrid='all_templates_both_directions';wide=meta.copy()
    for d in DOMAINS:
        a=df[allgrid,d,'target'];b=df[allgrid,d,'target_minus_control']
        for field in ('mean','lower','upper'):wide[f'{d}_{field}']=a[field]
        wide[f'{d}_control_mean']=a.matched_control_mean;wide[f'{d}_adjusted_mean']=b['mean'];wide[f'{d}_adjusted_lower']=b.lower;wide[f'{d}_adjusted_upper']=b.upper
        wide[f'{d}_positive_supported_both']=(a.lower>0)&(b.lower>0)
        wide[f'{d}_negative_supported_both']=(a.upper<0)&(b.upper<0)
    wide['positive_supported_domains']=wide.apply(lambda r:';'.join(d for d in DOMAINS if r[f'{d}_positive_supported_both']),axis=1)
    wide['negative_supported_domains']=wide.apply(lambda r:';'.join(d for d in DOMAINS if r[f'{d}_negative_supported_both']),axis=1)
    wide['positive_raw_means_all_three']=np.all(wide[[f'{d}_mean' for d in DOMAINS]].to_numpy()>0,axis=1)
    wide['positive_adjusted_means_all_three']=np.all(wide[[f'{d}_adjusted_mean' for d in DOMAINS]].to_numpy()>0,axis=1)
    wide.to_csv(P/'all_connections.csv',index=False)
    hrows=[]
    for h,sub in wide.groupby('sender',sort=False):
        # Select by equally weighting all three domain absolute means; descriptive only.
        scores=sub[[f'{d}_mean' for d in DOMAINS]].abs().mean(axis=1);best=sub.loc[scores.idxmax()]
        row=dict(sender=h,membership=best.membership,control=best.control,connections_tested=len(sub),strongest_observed_receiver=best.receiver,strongest_observed_component=best.component,aliases=best.aliases,
                 positive_supported_all_three=int((sub.positive_supported_domains=='C1;C2;K1').sum()),positive_raw_means_all_three=int(sub.positive_raw_means_all_three.sum()),positive_adjusted_means_all_three=int(sub.positive_adjusted_means_all_three.sum()))
        detected=est[(est.sender==h)&est.detected]
        row['connections_detected_any_cell']=len(detected[['receiver','component']].drop_duplicates())
        for d in DOMAINS:
            for stat in ('mean','control_mean','adjusted_mean'):row[f'{d}_{stat}']=best[f'{d}_{stat}']
            row[f'{d}_positive_supported_routes']=int(sub[f'{d}_positive_supported_both'].sum())
            row[f'{d}_negative_supported_routes']=int(sub[f'{d}_negative_supported_both'].sum())
        hrows.append(row)
    heads=pd.DataFrame(hrows);heads.to_csv(P/'all_41_heads.csv',index=False)
    bygroup=heads.groupby('membership',sort=False).agg(heads=('sender','size'),connections=('connections_tested','sum'),connections_detected_any_cell=('connections_detected_any_cell','sum'),shared_positive_supported=('positive_supported_all_three','sum'))
    bygroup.to_csv(P/'membership_summary.csv')
    summary=dict(seed=SEED,bootstrap_draws=DRAWS,alpha=ALPHA,cell_families=14,columns_per_cell=4436,exact_sign_assignments=1024,bootstrap_settings=settings,
        cell_detection_rows=int(est.detected.sum()),cell_detected_unique_connections=len(est[est.detected][['sender','receiver','component']].drop_duplicates()),
        full_grid_positive_patterns=wide.positive_supported_domains.value_counts().to_dict(),full_grid_negative_patterns=wide.negative_supported_domains.value_counts().to_dict(),
        positive_mean_all_three=int(wide.positive_raw_means_all_three.sum()),positive_adjusted_mean_all_three=int(wide.positive_adjusted_means_all_three.sum()),
        contrasts=contrasts.groupby('grid').agg(tested=('difference','size'),finite_bounds=('lower','count'),exclude_zero=('excludes_zero','sum')).to_dict('index'),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),validated_effects_sha256=hashlib.sha256((P/'validated_effects.npz').read_bytes()).hexdigest())
    (P/'statistics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='bootstrap_settings'},indent=2),flush=True)
    print('DETECTIONS\n',est[est.detected][['cell','sender','receiver','component','kind','mean','exact_maxstat_p_14_cells']].to_string(index=False))
    print('L39H23 QUERIES\n',wide[(wide.sender=='L39H23')&wide.receiver.isin(['L43H30','L45H38'])][['sender','receiver','component','C1_mean','C2_mean','K1_mean','C1_adjusted_mean','C2_adjusted_mean','K1_adjusted_mean','positive_supported_domains']].to_string(index=False))

if __name__=='__main__':main()
