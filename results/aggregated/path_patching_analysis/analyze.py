"""Analyze saved union41_v2 only. No model runs, raw-output edits, or route significance tests."""
from pathlib import Path
import hashlib
import json
import os
from collections import defaultdict
os.environ.setdefault('MPLCONFIGDIR','/tmp/path-analysis-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'results/circuit/path_patching/union41_v2'
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
study=read(DATA/'study.json'); plan=study['plan']
label=lambda h:'L%dH%d'%tuple(h)
assign={label(p['target']):{'control':label(p['control']),'membership':p['membership']} for p in plan['sender_pairs']}
selected=list(assign); controls=[assign[h]['control'] for h in selected]
assert len(selected)==len(set(controls))==41 and not set(selected)&set(controls)
source_checks={k:sha(DATA/'code_snapshot'/k)==v for k,v in study['source_sha256'].items()}
assert all(source_checks.values())
assert sha(DATA/'code_snapshot/results/aggregated/path_patching_setup/plan.json')==study['plan_sha256']
paths=[];whole=[];audits=[];bad=[];hashes={};cell_meta={};baselines=[]
all_zero_delta=0
for spec in plan['cells']:
 cell=spec['id'];folder=DATA/cell;cell_meta[cell]=spec
 pairfile=folder/'pairs.jsonl';pairs=[json.loads(x) for x in pairfile.read_text().splitlines()]
 assert len(pairs)==study['pairs_per_cell']==10
 assert read(folder/'pairs.meta.json')['pairs_sha256']==sha(pairfile)
 task=read(folder/'task.json'); assert task['plan_sha256']==study['plan_sha256']
 assert task['study_script_sha256']==study['source_sha256']['scripts/circuit/path_patching_study.py']
 for scope in ('existing_candidates','exploratory_layer47'):
  sub=folder/scope;identity=read(sub/'manifest.json')['identity']
  assert identity['pairs_sha256']==sha(pairfile)
  assert identity['script_sha256']==study['source_sha256']['scripts/circuit/path_patching.py']
  assert identity['versions']==study['versions']
  assert identity['arguments']['revision']==plan['revision']
  files=sorted(sub.glob('pair_*.json')); assert len(files)==10
  expected_senders={label(p[k]) for p in plan['sender_pairs'] for k in ('target','control')
                    if (p['target'][0]==46)==(scope=='exploratory_layer47')}
  meta=None;effect=[];norm=[];errs=defaultdict(float);times=[];drift_pairs=0;zero_rows=0
  for i,file in enumerate(files):
   hashes[str(file.relative_to(ROOT))]=sha(file);saved=read(file)
   assert saved['pair']==pairs[i];rows=saved['rows'];times.append(saved['measurement_seconds'])
   assert len(rows)==(4212 if scope=='existing_candidates' else 224)
   assert {r['senders'] for r in rows}==expected_senders
   keys=[(r['senders'],r['component'],r['routes'][0]['layer'],r['routes'][0]['hook_index']) for r in rows]
   assert len(set(keys))==len(keys)
   if meta is None:meta=keys
   assert keys==meta
   vals=np.array([r['path_effect'] for r in rows],dtype=float);assert np.isfinite(vals).all()
   norms=np.array([r['receiver_delta_l2'][0] for r in rows]);effect.append(vals);norm.append(norms)
   assert np.allclose(vals,[r['path_margin']-r['batch_clean_margin'] for r in rows],rtol=0,atol=1e-10)
   seen={}
   for row in rows:
    h=row['senders'];w=row['whole_sender_effect']
    assert w==row['whole_sender_margin']-row['clean_margin']
    assert h not in seen or seen[h]==w;seen[h]=w
    for k,v in row.items():
     if k.endswith('max_logit_error'):errs[k]=max(errs[k],v)
    if row['receiver_delta_l2']==[0.0]:
     all_zero_delta+=1;zero_rows+=1
     if row['path_effect']!=0:
      bad.append({'file':str(file.relative_to(ROOT)),'cell':cell,'pair':i,**{k:row[k] for k in
       ('senders','receivers','component','path_effect','clean_margin','batch_clean_margin','receiver_delta_l2')}})
   for h,w in seen.items():
    whole.append({'cell':cell,'domain':spec['condition'],'template':spec['template'],'base_rule':spec['base_rule'],
                  'pair':i,'sender':h,'effect':w,'scope':scope})
   max_margin_drift=max(abs(r['batch_clean_margin']-r['clean_margin']) for r in rows)
   drift_pairs+=max_margin_drift>0
   baselines.append({'cell':cell,'scope':scope,'pair':i,'clean_margin':rows[0]['clean_margin'],
                     'max_batch_clean_margin_drift':max_margin_drift,
                     'batch_vs_single_max_logit_error':rows[0]['batch_vs_single_max_logit_error']})
  x=np.array(effect);n=np.array(norm);assert x.shape[0]==10
  for j,(h,c,l,index) in enumerate(meta):
   paths.append({'cell':cell,'domain':spec['condition'],'scope':scope,'sender':h,'component':c,
                 'receiver_layer':l,'receiver_index':index,'receiver_id':f'L{l}'+('H' if c=='q' else 'KV')+str(index),
                 'mean_path':x[:,j].mean(),'sd_path':x[:,j].std(ddof=1),'positive_count':int((x[:,j]>0).sum()),
                 'negative_count':int((x[:,j]<0).sum()),'n':10,'mean_receiver_delta_l2':n[:,j].mean()})
  audits.append({'cell':cell,'scope':scope,'pairs':len(files),'rows':int(x.size),'seconds':sum(times),
                 'pairs_with_batch_margin_drift':drift_pairs,'zero_delta_rows':zero_rows,**errs})
  for check in ('sender_self_max_logit_error','receiver_noop_max_logit_error','self_path_max_logit_error','batch_row_max_logit_error'):
   assert errs[check]<=identity['arguments']['tolerance']
whole=pd.DataFrame(whole);paths=pd.DataFrame(paths);audits=pd.DataFrame(audits);baseline=pd.DataFrame(baselines)
assert len(whole)==82*140 and len(paths)==4436*14
whole.to_csv(OUT/'whole_sender_per_prompt.csv',index=False)
paths.to_csv(OUT/'path_cell_descriptive.csv',index=False)
audits.to_csv(OUT/'coverage_and_numerics.csv',index=False)
baseline.to_csv(OUT/'clean_margin_diagnostics.csv',index=False)
pd.DataFrame(bad).to_csv(OUT/'zero_change_nonzero_path.csv',index=False)

# Bootstrap prompt rows jointly, preserving dependence among heads and target/control.
# Bonferroni across cells; linear interval propagation needs no independence between cells.
rng=np.random.default_rng(20261022);draws=50000;alpha=.05/14
cell_bounds=[];boot_info=[]
for cell in cell_meta:
 pivot=whole[whole.cell==cell].pivot(index='pair',columns='sender',values='effect')
 targets=pivot[selected].to_numpy();ctrl=pivot[controls].to_numpy()
 x=np.concatenate([targets,ctrl,targets-ctrl],axis=1)
 mean=x.mean(0);se=x.std(0,ddof=1)/np.sqrt(10);valid=se>0
 weights=rng.multinomial(10,np.full(10,.1),size=draws)/10
 max_error=np.max(np.abs((weights@x[:,valid]-mean[valid])/se[valid]),axis=1)
 critical=float(np.quantile(max_error,1-alpha))
 lower=np.where(valid,mean-critical*se,np.nan);upper=np.where(valid,mean+critical*se,np.nan)
 for j in range(123):
  kind=('target','control','target_minus_control')[j//41];h=selected[j%41]
  cell_bounds.append({'cell':cell,'domain':cell_meta[cell]['condition'],'sender':h,'kind':kind,
                      'mean':mean[j],'lower':lower[j],'upper':upper[j],'standard_error':se[j]})
 boot_info.append({'cell':cell,'critical':critical,'zero_variance_series':int((~valid).sum())})
bounds=pd.DataFrame(cell_bounds);bounds.to_csv(OUT/'whole_sender_cell_bounds.csv',index=False)
def mean_all(v):return np.mean(v.to_numpy())  # keep NaN for unestimated uncertainty
aggregate=bounds.groupby(['domain','sender','kind'],sort=False).agg(mean=('mean','mean'),lower=('lower',mean_all),upper=('upper',mean_all)).reset_index()
aggregate.to_csv(OUT/'whole_sender_domain_bounds.csv',index=False)
contrasts=[]
for h in selected:
 for kind in ('target','target_minus_control'):
  for left,right in [('C1','C2'),('C1','K1'),('C2','K1')]:
   a=aggregate[(aggregate.domain==left)&(aggregate.sender==h)&(aggregate.kind==kind)].iloc[0]
   b=aggregate[(aggregate.domain==right)&(aggregate.sender==h)&(aggregate.kind==kind)].iloc[0]
   contrasts.append({'sender':h,'membership':assign[h]['membership'],'kind':kind,'left':left,'right':right,
                     'difference':a['mean']-b['mean'],'lower':a.lower-b.upper,'upper':a.upper-b.lower})
contrasts=pd.DataFrame(contrasts);contrasts.to_csv(OUT/'whole_sender_domain_contrasts.csv',index=False)
summary=[]
for h in selected:
 row={'sender':h,**assign[h]}
 for dom in ('C1','C2','K1'):
  for kind in ('target','control','target_minus_control'):
   a=aggregate[(aggregate.domain==dom)&(aggregate.sender==h)&(aggregate.kind==kind)].iloc[0]
   for stat in ('mean','lower','upper'):row[f'{dom}_{kind}_{stat}']=a[stat]
 summary.append(row)
summary=pd.DataFrame(summary);summary.to_csv(OUT/'all_41_heads.csv',index=False)
# Same-route comparisons, descriptive only. Never sum separate routes into a recovery fraction.
routekeys=['domain','scope','sender','component','receiver_layer','receiver_index','receiver_id']
pa=paths.groupby(routekeys,sort=False).agg(mean_path=('mean_path','mean'),positive_count=('positive_count','sum'),negative_count=('negative_count','sum'),n=('n','sum'),mean_receiver_delta_l2=('mean_receiver_delta_l2','mean')).reset_index()
pa.to_csv(OUT/'path_domain_descriptive.csv',index=False)
targets=pa[pa.sender.isin(selected)].copy();targets['control_sender']=targets.sender.map(lambda h:assign[h]['control'])
matchkeys=['domain','scope','component','receiver_layer','receiver_index']
ct=pa.rename(columns={'sender':'control_sender','mean_path':'control_mean_path'})
targets=targets.merge(ct[matchkeys+['control_sender','control_mean_path']],on=matchkeys+['control_sender'],validate='many_to_one')
targets['target_minus_control']=targets.mean_path-targets.control_mean_path
targets.to_csv(OUT/'path_target_control_descriptive.csv',index=False)
best=targets.loc[targets.groupby(['domain','sender','component']).mean_path.idxmax()]
best.to_csv(OUT/'strongest_observed_routes_NOT_confirmed.csv',index=False)

# Whole intervention comparison with previously saved CMA profiles; no identical-cohort assumption.
old=pd.read_csv(ROOT/'results/aggregated/head_domain_comparison/cma_cell_profiles.csv')
old=old[old.sensitivity.astype(str).str.lower().eq('false') & old.condition.isin(['C1','C2','K1'])]
oldagg=old.groupby(['condition','head_name']).mean_cma.mean()
corr=[]
for dom in ('C1','C2','K1'):
 new=summary.set_index('sender')[f'{dom}_target_mean'];prior=oldagg.loc[dom].reindex(new.index)
 corr.append({'domain':dom,'pearson_r_41_head_means':float(np.corrcoef(new,prior)[0,1])})
pd.DataFrame(corr).to_csv(OUT/'comparison_to_previous_cma.csv',index=False)

# Scientific figure: all heads, actual signed values. No significance markers on paths.
fig,axes=plt.subplots(1,2,figsize=(10,15),sharey=True,layout='constrained')
short={'belief':'B','photo':'P','code':'K','belief + photo':'BP','belief + code':'BK','belief + photo + code':'BPK'}
labels=[f"{h} [{short[assign[h]['membership']]}]" for h in selected]
for ax,kind,title in zip(axes,['target','target_minus_control'],['Whole-head effect','Whole-head effect minus matched control']):
 matrix=summary[[f'{d}_{kind}_mean' for d in ('C1','C2','K1')]].to_numpy()
 im=ax.imshow(matrix,cmap='RdBu_r',vmin=-6.5,vmax=6.5,aspect='auto')
 for i in range(41):
  for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>3.5 else 'black')
 ax.set_xticks(range(3),['Belief','Photo','Code']);ax.set_yticks(range(41),labels);ax.tick_params(axis='y',labelsize=9);ax.set_title(title,fontsize=11)
fig.colorbar(im,ax=axes,shrink=.3,label='Change in causal − original answer logit margin')
fig.suptitle('All 41 heads: whole-head interventions on the new prompts\nB/P/K labels indicate original selection membership; 60 story pairs/domain, 20 code pairs',fontsize=12)
fig.savefig(OUT/'all_41_whole_head_effects.png',dpi=180);plt.close(fig)
info={'study':'union41_v2','checkpoint_files':len(hashes),'path_rows':int(audits.rows.sum()),'unique_prompt_pairs':140,
      'source_hash_checks':source_checks,'source_checkpoints':hashes,'zero_delta_rows':all_zero_delta,
      'zero_delta_nonzero_path_rows':len(bad),'whole_bootstrap':{'draws':draws,'seed':20261022,'family':'123 whole-head/control series jointly per cell; alpha=.05/14; signed interval propagation across fixed cells','cell_info':boot_info,'zero_variance':'uncertainty unavailable (NaN), not zero uncertainty'},
      'path_inference':'descriptive only: missing batched clean-receiver transplant; nine zero-change nonzero effects observed',
      'analysis_script_sha256':sha(Path(__file__))}
(OUT/'audit.json').write_text(json.dumps(info,indent=2)+'\n')
print('COMPLETED',len(hashes),'checkpoints;',audits.rows.sum(),'path rows; whole measurements',len(whole))
print('CMA correlations',corr)
print('FOUR BELIEF CANDIDATES, RAW WHOLE CONTRASTS')
print(contrasts[(contrasts.sender.isin(['L31H17','L31H21','L32H26','L33H1']))&(contrasts.kind=='target')&(contrasts.left=='C1')].to_string(index=False))
print('RAW CONTRAST BOUNDS EXCLUDING ZERO')
print(contrasts[(contrasts.kind=='target')&((contrasts.lower>0)|(contrasts.upper<0))].to_string(index=False))
print('Whole bootstrap and all 41-head tables saved in',OUT)
