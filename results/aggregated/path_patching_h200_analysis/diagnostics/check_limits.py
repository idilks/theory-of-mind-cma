"""Existing-data sensitivity diagnostics; no power claim or new model runs.
Uncorrected tests are diagnostic only. Global summaries average separate path
interventions; they are NOT a simultaneous intervention or recovered fraction.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).parent; BASE=P.parent; ROOT=BASE.resolve().parents[2]
sys.path.insert(0,str(BASE))
from analyze import simultaneous_cell
meta=pd.read_csv(BASE/'connections.csv');z=np.load(BASE/'validated_effects.npz');cells=pd.read_csv(BASE/'control_audit_by_cell.csv')
sign=2*((np.arange(1024)[:,None]>>np.arange(10))&1)-1
summary=[];details=[];examples=[];aggregate=[];whole=[];resolution=[]
keys=('target','target_minus_control');rng=np.random.default_rng(20261024)
for row in cells.itertuples():
 cid=row.cell;x=z[cid+'__target'];ctrl=z[cid+'__control'];y=np.concatenate([x,x-ctrl],1)
 means=y.mean(0);sd=y.std(0,ddof=1);se=sd/np.sqrt(10)
 draws=abs(sign@y/10); maxima=draws.max(1)
 p0=(draws>=abs(means)[None,:]-1e-12).mean(0)
 pmax=(maxima[:,None]>=abs(means)[None,:]-1e-12).mean(0)
 pfinal=np.minimum(1,14*pmax)
 nonzero=(y!=0).sum(0);pos=(y>0).sum(0);neg=(y<0).sum(0)
 floor=np.minimum(1,14*2.**(1-nonzero))
 assert np.all(pfinal+1e-12>=floor)
 assert np.all((pfinal>.05)|((pos==10)|(neg==10)))
 for j,kind in enumerate(keys):
  s=slice(j*2218,(j+1)*2218);f=meta.copy();f['cell']=cid;f['kind']=kind;f['mean']=means[s];f['sd']=sd[s];f['se']=se[s]
  f['positive']=pos[s];f['negative']=neg[s];f['nonzero']=nonzero[s];f['minimum_p_from_nonzero_count']=floor[s]
  f['pointwise_p_DIAGNOSTIC']=p0[s];f['within_cell_max_p']=pmax[s];f['full_study_p']=pfinal[s];details.append(f)
  summary.append(dict(cell=cid,kind=kind,route_cells=2218,zero_prompt_effect_fraction=float((y[:,s]==0).mean()),
   median_abs_mean=float(np.median(abs(means[s]))),median_sd=float(np.median(sd[s])),median_se=float(np.median(se[s])),
   all_ten_same_nonzero_sign=int(((pos[s]==10)|(neg[s]==10)).sum()),
   impossible_due_to_zeros=int((floor[s]>.05).sum()),pointwise_p05=int((p0[s]<=.05).sum()),within_cell_max_p05=int((pmax[s]<=.05).sum()),full_study_p05=int((pfinal[s]<=.05).sum())))
 # Averages over 39 main-study senders, excluding the two exploratory L46 senders.
 mask=meta.scope=='existing_candidates'; vectors=[];names=[]
 for comp in ('all','q','k','v'):
  ix=mask & ((meta.component==comp) if comp!='all' else True)
  assert meta[ix].sender.nunique()==39
  for kind,vals in [('target',x),('control',ctrl),('target_minus_control',x-ctrl)]:
   names.append(comp+'__'+kind);vectors.append(vals[:,ix].mean(1))
 q=(x-ctrl)[:,mask&(meta.component=='q')].mean(1);v=(x-ctrl)[:,mask&(meta.component=='v')].mean(1)
 names.append('q_minus_v__target_minus_control');vectors.append(q-v)
 g=np.column_stack(vectors);mu,lo,hi,p,crit,nzero=simultaneous_cell(g,rng)
 for j,name in enumerate(names):
  aggregate.append(dict(cell=cid,domain=row.domain,quantity=name,mean=mu[j],lower=lo[j],upper=hi[j],p_corrected=p[j],positive_prompts=int((g[:,j]>0).sum()),negative_prompts=int((g[:,j]<0).sum())))
 # Compare whole-head and path effects on the SAME selected repaired dataset.
 folder=ROOT/row.path_root/cid
 for scope in ('existing_candidates','exploratory_layer47'):
  for file in sorted((folder/scope).glob('pair_*.json')):
   data=json.loads(file.read_text());seen={}
   for rr in data['rows']:
    h=rr['senders'];wh=rr['whole_sender_effect']
    if h in seen:assert seen[h]==wh
    else:seen[h]=wh
   for m in meta[meta.scope==scope].drop_duplicates('sender').itertuples():
    whole.append(dict(cell=cid,domain=row.domain,pair=data['pair']['pair_id'],sender=m.sender,control=m.control,membership=m.membership,target=seen[m.sender],matched_control=seen[m.control],target_minus_control=seen[m.sender]-seen[m.control]))
 arr=x.ravel();nz=abs(arr[arr!=0]);resolution.append(dict(cell=cid,min_nonzero=float(nz.min()),median_nonzero=float(np.median(nz)),fraction_multiples_one_128=float(np.mean(np.isclose(arr*128,np.round(arr*128),rtol=0,atol=1e-12)))))
 for sender,recv in [('L39H23','L43H30'),('L39H23','L45H38')]:
  i=np.flatnonzero((meta.sender==sender)&(meta.receiver==recv)&(meta.component=='q'))[0]
  for j,kind in enumerate(keys):
   idx=i+j*2218;vals=y[:,idx]
   examples.append(dict(cell=cid,sender=sender,receiver=recv,kind=kind,mean=means[idx],sd=sd[idx],se=se[idx],minimum_p=floor[idx],pointwise_p=p0[idx],within_cell_max_p=pmax[idx],full_study_p=pfinal[idx],leave_one_out_min=float(((vals.sum()-vals)/9).min()),leave_one_out_max=float(((vals.sum()-vals)/9).max()),values=json.dumps(vals.tolist())))
pd.DataFrame(summary).to_csv(P/'detection_limits_by_cell.csv',index=False)
detail=pd.concat(details);detail.to_csv(P/'all_route_diagnostics.csv',index=False)
pd.DataFrame(examples).to_csv(P/'candidate_diagnostics.csv',index=False)
pd.DataFrame(resolution).to_csv(P/'observed_resolution.csv',index=False)
g=pd.DataFrame(aggregate);g.to_csv(P/'global_separate_intervention_averages.csv',index=False)
wh=pd.DataFrame(whole);wh.to_csv(P/'whole_head_effects_validated.csv',index=False)
wh.groupby(['sender','membership','domain'])[['target','matched_control','target_minus_control']].mean().to_csv(P/'whole_head_means.csv')
# Preserve unknown cross-cell dependence: same endpoint-averaging method as prior report.
glob=[]
for (domain,quantity),gg in g.groupby(['domain','quantity']):
 glob.append(dict(domain=domain,quantity=quantity,mean=gg['mean'].to_numpy().mean(),lower=gg.lower.to_numpy().mean(),upper=gg.upper.to_numpy().mean()))
for quantity,gg in pd.DataFrame(glob).groupby('quantity'):
 glob.append(dict(domain='equal_domain_average',quantity=quantity,mean=gg['mean'].to_numpy().mean(),lower=gg.lower.to_numpy().mean(),upper=gg.upper.to_numpy().mean()))
pd.DataFrame(glob).to_csv(P/'global_domain_averages.csv',index=False)
# Quantify how the previous construction uses the 14 cells.
st=json.loads((BASE/'statistics.json').read_text());critical=[r['bootstrap_critical'] for r in st['bootstrap_settings']]
report={'minimum_two_sided_p_at_ten_all_nonzero':2/1024,'minimum_14cell_corrected_p_nine_nonzero':14*2/512,
 'cell_bootstrap_critical_min':min(critical),'cell_bootstrap_critical_max':max(critical),
 'by_kind':{},'input_sha256':hashlib.sha256((BASE/'validated_effects.npz').read_bytes()).hexdigest()}
for k,t in pd.DataFrame(summary).groupby('kind'):
 report['by_kind'][k]={f:int(t[f].sum()) for f in ['route_cells','all_ten_same_nonzero_sign','impossible_due_to_zeros','pointwise_p05','within_cell_max_p05','full_study_p05']}
 report['by_kind'][k]['zero_prompt_effect_fraction']=float(t.zero_prompt_effect_fraction.mean())
(P/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print('GLOBAL AVERAGES',pd.DataFrame(glob).to_string(index=False))
print('EXAMPLES',pd.DataFrame(examples).query("cell in ['C2_library_book_ABA','K1_indexed_list_ABB']").to_string(index=False))
print('WHOLE HEAD',wh.query("sender in ['L29H31','L35H30']").groupby(['sender','domain'])[['target','target_minus_control']].mean().to_string())

# Reproducible illustration of interval conservatism (not an independence claim).
contrast=pd.read_csv(BASE/'domain_differences.csv');out=[]
for receiver in ['L43H30','L45H38']:
 ix=np.flatnonzero((meta.sender=='L39H23')&(meta.receiver==receiver)&(meta.component=='q'))[0]
 for kind in keys:
  variances={}
  for dom in ('C1','C2','K1'):
   ids=cells.loc[cells.domain==dom,'cell'].tolist();values=[]
   for cid in ids:
    vals=z[cid+'__target'][:,ix]
    if kind=='target_minus_control':vals=vals-z[cid+'__control'][:,ix]
    values.append(vals.var(ddof=1)/10)
   variances[dom]=sum(values)/len(ids)**2
  con=contrast[(contrast.grid=='all_templates_both_directions')&(contrast.sender=='L39H23')&(contrast.receiver==receiver)&(contrast.component=='q')&(contrast.kind==kind)]
  for r in con.itertuples():
   se=np.sqrt(variances[r.left]+variances[r.right]);out.append(dict(sender='L39H23',receiver=receiver,kind=kind,left=r.left,right=r.right,difference=r.difference,previous_lower=r.lower,previous_upper=r.upper,standard_error_IF_cells_independent=se,previous_halfwidth_over_independent_se=(r.upper-r.lower)/2/se))
pd.DataFrame(out).to_csv(P/'domain_interval_conservatism.csv',index=False)
steering_source=ROOT/'results/circuit/stage2_steering/Qwen2.5-14B-Instruct/L41_main/summary.csv'
s=pd.read_csv(steering_source)
s=s[(s.prompt_set=='true_belief')&((s['mode']=='clean')|((s['mode']=='inject')&(s.dose==2)&s.direction.isin(['target_rank1','random','shuffled'])))]
s.to_csv(P/'existing_steering_example.csv',index=False)
report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report['existing_steering_summary_sha256']=hashlib.sha256(steering_source.read_bytes()).hexdigest()
(P/'summary.json').write_text(json.dumps(report,indent=2))
