"""Independent checks of statistical boundaries, saved source hashes and summaries."""
import itertools
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import simultaneous_cell, mean_bounds

P=Path(__file__).parent
ROOT=P.resolve().parents[2]
# Exact test: a constant positive series has only all-plus/all-minus extrema.
y=np.column_stack([np.ones(10),np.zeros(10)])
m,lo,hi,p,*_=simultaneous_cell(y,np.random.default_rng(0),draws=1000)
assert p[0]==28/1024 and p[1]==1 and np.isnan(lo).all() and np.isnan(hi).all()
# Sign inversion flips bounds/means and leaves exact two-sided p unchanged.
y=np.column_stack([np.arange(10),np.arange(10)**2])
a=simultaneous_cell(y,np.random.default_rng(1),draws=1000)
b=simultaneous_cell(-y,np.random.default_rng(1),draws=1000)
assert np.allclose(a[0],-b[0]) and np.allclose(a[1],-b[2]) and np.array_equal(a[3],b[3])
# A missing cell bound must never disappear during averaging.
t1=pd.DataFrame({'mean':[1.], 'lower':[np.nan], 'upper':[np.nan]})
t2=pd.DataFrame({'mean':[3.], 'lower':[2.], 'upper':[4.]})
bounds=mean_bounds([t1,t2]);assert bounds['mean'][0]==2 and np.isnan(bounds['lower'][0])
print('Statistical boundary checks passed')

au=json.loads((P/'audit.json').read_text())
for path,digest in au['source_hashes'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
print('Unchanged audited source files:',len(au['source_hashes']))
meta=pd.read_csv(P/'connections.csv');z=np.load(P/'validated_effects.npz')
cells=pd.read_csv(P/'control_audit_by_cell.csv');est=pd.read_csv(P/'cell_estimates.csv');wide=pd.read_csv(P/'all_connections.csv')
assert cells.passed.dtype==bool and cells.passed.all()
assert len(est)==14*2218*2 and len(wide)==2218 and wide.sender.nunique()==41
assert len(meta[meta.scope=='existing_candidates'])==39*54
assert len(meta[meta.scope=='exploratory_layer47'])==2*56
# Old per-cell means must match precisely for every untouched path arm.
old=pd.read_csv(P.parent/'path_patching_clean_analysis/all_connection_cell_means.csv')
new=pd.read_csv(P/'cell_means.csv')
for row in cells.itertuples():
    if not row.path_repaired:
        aa=old[old.cell==row.cell].reset_index(drop=True)
        bb=new[new.cell==row.cell].reset_index(drop=True)
        for k in ('target_mean','control_mean','target_minus_control_mean'):
            assert np.allclose(aa[k],bb[k],rtol=0,atol=1e-14)
# Independent scalar enumeration for the detected code connection, with the
# entire family maximum recomputed per sign assignment (no vectorized p routine).
y=np.concatenate([z['K1_indexed_list_ABB__target'],z['K1_indexed_list_ABB__target']-z['K1_indexed_list_ABB__control']],axis=1)
ix=np.flatnonzero((meta.sender=='L39H23')&(meta.receiver=='L43H30')&(meta.component=='q'))[0]
obs=abs(y[:,2218+ix].mean());hits=0
for signs in itertools.product((-1,1),repeat=10):
    hits += np.max(abs((y*np.array(signs)[:,None]).mean(0)))>=obs-1e-12
assert hits==2 and hits/1024*14==.02734375
assert np.all(y[:,2218+ix]>0)
# Direct domain mean calculation from all saved prompt effects agrees with table.
for d in ('C1','C2','K1'):
    ids=cells.loc[cells.domain==d,'cell'].tolist()
    raw=np.concatenate([z[c+'__target'] for c in ids]).mean(0)
    ctr=np.concatenate([z[c+'__control'] for c in ids]).mean(0)
    assert np.allclose(wide[d+'_mean'],raw,rtol=0,atol=1e-14)
    assert np.allclose(wide[d+'_adjusted_mean'],raw-ctr,rtol=0,atol=1e-14)
contrasts=pd.read_csv(P/'domain_differences.csv')
assert len(contrasts)==6*3*2*2218 and contrasts.excludes_zero.dtype==bool
assert not contrasts.excludes_zero.any()
print('Verified: unchanged path arms, all 41 senders, 2218 routes, exact corrected p=0.02734375, domain means and all contrasts')
