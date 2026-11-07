"""Audit clean transplants and compare all 41 senders. Saved outputs only; no inference."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/path-clean-analysis-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import sys, json, hashlib, re
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
SOURCE = ROOT/'results/circuit/path_patching/union41_v2'
CLEAN = ROOT/'results/circuit/path_patching/union41_v2_clean_control_v1'
sys.path.insert(0, str(ROOT/'codebase/tasks/causal_analysis'))
from statistical_testing import permutation_test_heads
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
label = lambda h: 'L%dH%d' % tuple(h)
study, controlstudy = read(SOURCE/'study.json'), read(CLEAN/'study.json')
plan = study['plan']; tol = controlstudy['tolerance']
assert controlstudy['source_study_sha256'] == sha(SOURCE/'study.json')
assert controlstudy['source_plan'] == plan
assert controlstudy['versions'] == study['versions']
assert controlstudy['control_script_sha256'] == sha(CLEAN/'code_snapshot/scripts/circuit/path_patching_clean_control.py')
assert all(sha(SOURCE/'code_snapshot'/p)==h for p,h in study['source_sha256'].items())
assign = {label(p['target']): dict(control=label(p['control']), membership=p['membership']) for p in plan['sender_pairs']}
assert len(assign)==41
hashes = {}; cellrows = []; detailed_controls = []; zero_cases = []; arrays = {}; metadata = None
primary = {'C1':'C1_library_book_ABA','C2':'C2_library_book_ABA','K1':'K1_indexed_list_ABA'}

for taskid, cell in enumerate(plan['cells']):
    cid = cell['id']; rawfolder=SOURCE/cid; folder=CLEAN/cid
    manifest=read(folder/'manifest.json'); runtime=read(folder/'runtime.json')
    assert runtime['provenance']['versions']==study['versions']
    assert all(sha(SOURCE/p)==h for p,h in manifest['source_sha256'].items())
    hashes[str(folder.relative_to(ROOT)/'manifest.json')]=sha(folder/'manifest.json')
    task=read(rawfolder/'task.json')['task']
    assert sum(len(g['sender_pairs']) for g in task['groups'])==41
    pairs=[json.loads(s) for s in (rawfolder/'pairs.jsonl').read_text().splitlines() if s]
    assert len(pairs)==10 and sha(rawfolder/'pairs.jsonl')==read(rawfolder/'pairs.meta.json')['pairs_sha256']
    original_log=ROOT/f'slurm_logs/path_union41_9923275_{taskid}.out'
    line=original_log.read_text().splitlines()[0]
    hostname=line.split(' on ')[1]
    matrix=[]; ctrlmatrix=[]; cellmeta: list[dict]=[]
    failures=nonzero=mappedfail=single_drift=batchdrift=0
    maxeffect=maxerror=0.; seconds=0.; cellsingle=set(); cellbatch=set()
    allsame=True; frozenerror=0.; totalrows=0
    for group in task['groups']:
        scope=group['scope']; xs=[]; cs=[]; groupmeta=None
        files=sorted((folder/scope).glob('pair_*.json')); assert len(files)==10
        for i,file in enumerate(files):
            saved=read(file); rawfile=rawfolder/scope/file.name; raw=read(rawfile)
            hashes[str(file.relative_to(ROOT))]=sha(file)
            assert saved['source_sha256']==sha(rawfile)
            assert saved['pair']==raw['pair']==pairs[i]
            seconds+=saved['measurement_seconds']
            allsame &= saved['clean_and_frozen_receiver_caches_identical']
            frozenerror=max(frozenerror,saved['frozen_clean_max_logit_error'])
            if saved['single_clean_margin_drift_from_original'] != 0: cellsingle.add(i)
            assert saved['single_clean_margin_drift_from_original']==saved['single_clean_margin']-raw['rows'][0]['clean_margin']
            mapping={}
            for cr in saved['rows']:
                fields=['control_effect','max_logit_error','clean_cache_control_effect','clean_cache_max_logit_error',
                        'batch_row_max_logit_error','untouched_reference_max_logit_error']
                passed=max([abs(cr[k]) for k in fields]+[saved['frozen_clean_max_logit_error']])<=tol
                assert passed==cr['passes_tolerance']
                failures+=not passed; nonzero+=cr['control_effect']!=0
                maxeffect=max(maxeffect,abs(cr['control_effect'])); maxerror=max(maxerror,cr['max_logit_error'])
                orig_margins=[]
                for ref in cr['source_rows']:
                    idx=ref['row_index']; assert idx not in mapping; rr=raw['rows'][idx]
                    assert ref['sender']==rr['senders']
                    assert all(rr[k]==cr[k] for k in ('receivers','receiver_aliases','component','routes'))
                    # Exact original row position within each sender's intervention batches.
                    per_sender=54 if scope=='existing_candidates' else 56
                    ordinal=idx%per_sender
                    assert cr['batch_row']==ordinal%study['batch_size']+1
                    assert cr['batch_width']==min(study['batch_size'],per_sender-ordinal//study['batch_size']*study['batch_size'])+1
                    mapping[idx]=cr; mappedfail+=not passed; orig_margins.append(rr['batch_clean_margin'])
                    if rr['receiver_delta_l2']==[0.] and rr['path_effect']!=0:
                        zero_cases.append(dict(cell=cid,scope=scope,pair=i,sender=rr['senders'],receiver=rr['receivers'],component=rr['component'],
                            original_effect=rr['path_effect'],new_clean_effect=cr['control_effect'],original_host=hostname,control_gpu=runtime['gpu']))
                assert sorted(set(orig_margins))==cr['original_batch_clean_margins']
                drift=any(m!=cr['batch_clean_margin'] for m in orig_margins)
                batchdrift+=drift
                if drift: cellbatch.add(i)
                detailed_controls.append(dict(cell=cid,scope=scope,pair=i,component=cr['component'],receiver=cr['receivers'],
                    batch_width=cr['batch_width'],batch_row=cr['batch_row'],passes=passed,
                    control_effect=cr['control_effect'],max_logit_error=cr['max_logit_error'],
                    original_margin_drift=drift,mapped_rows=len(cr['source_rows'])))
            assert set(mapping)==set(range(len(raw['rows'])))
            assert len(raw['rows'])==(4212 if scope=='existing_candidates' else 224)
            totalrows+=len(raw['rows'])
            lookup={}; metas=[]; row_effects=[]; row_controls=[]
            for idx,rr in enumerate(raw['rows']):
                route=rr['routes'][0]; assert len(rr['routes'])==1
                key=(rr['senders'],rr['component'],route['layer'],route['hook_index'])
                assert key not in lookup
                assert rr['path_effect']==rr['path_margin']-rr['batch_clean_margin']
                lookup[key]=rr['path_effect']
            for rr in raw['rows']:
                if rr['senders'] not in assign: continue
                h=rr['senders']; route=rr['routes'][0]; comp=rr['component']; l=route['layer']; ix=route['hook_index']
                metas.append(dict(sender=h,**assign[h],scope=scope,component=comp,receiver_layer=l,receiver_index=ix,
                    receiver=f'L{l}'+('H' if comp=='q' else 'KV')+str(ix), aliases=';'.join(rr['receiver_aliases'])))
                row_effects.append(rr['path_effect']);row_controls.append(lookup[(assign[h]['control'],comp,l,ix)])
            if groupmeta is None:groupmeta=metas
            assert metas==groupmeta
            xs.append(row_effects);cs.append(row_controls)
        assert groupmeta is not None
        matrix.append(np.array(xs));ctrlmatrix.append(np.array(cs));cellmeta+=groupmeta
    x=np.concatenate(matrix,axis=1);c=np.concatenate(ctrlmatrix,axis=1)
    assert x.shape==c.shape==(10,2218) and np.isfinite(x).all() and np.isfinite(c).all()
    if metadata is None:metadata=cellmeta
    assert metadata==cellmeta
    arrays[cid]=(x,c)
    same_family=hostname.startswith('h200') and 'H200' in runtime['gpu']
    compatible=(failures==0 and not cellsingle and not cellbatch and same_family)
    cellrows.append(dict(cell=cid,domain=cell['condition'],original_host=hostname,control_gpu=runtime['gpu'],
        control_rows=1100,failed_controls=failures,nonzero_answer_effects=nonzero,max_abs_control_effect=maxeffect,
        max_logit_error=maxerror,mapped_original_rows=totalrows,mapped_failed_rows=mappedfail,
        pairs_with_single_margin_drift=len(cellsingle),pairs_with_batch_margin_drift=len(cellbatch),
        control_rows_with_batch_margin_drift=batchdrift,clean_frozen_caches_equal=allsame,frozen_clean_max_logit_error=frozenerror,
        compatible=compatible,measurement_seconds=seconds))
    print(cid, 'controls failed', failures, 'baseline-drift pairs', len(cellsingle|cellbatch), 'compatible',compatible,flush=True)

audit=pd.DataFrame(cellrows);audit.to_csv(OUT/'control_audit_by_cell.csv',index=False)
pd.DataFrame(detailed_controls).to_csv(OUT/'clean_controls.csv',index=False)
pd.DataFrame(zero_cases).to_csv(OUT/'original_zero_change_counterexamples.csv',index=False)
assert audit.control_rows.sum()==15400 and audit.mapped_original_rows.sum()==621040
assert all(audit.set_index('cell').loc[cid,'compatible'] for cid in primary.values())
meta=pd.DataFrame(metadata);meta.to_csv(OUT/'connections.csv',index=False)
# Complete original-domain means are descriptive, including cells not validated by this follow-up.
full_frames=[]; cellmeans=[]
for cell in plan['cells']:
    x,c=arrays[cell['id']]
    frame=meta.copy();frame['cell']=cell['id'];frame['domain']=cell['condition'];frame['target_mean']=x.mean(0);frame['control_mean']=c.mean(0)
    frame['target_minus_control_mean']=(x-c).mean(0)
    frame['compatible_cell']=bool(audit.set_index('cell').loc[cell['id'],'compatible'])
    cellmeans.append(frame)
pd.concat(cellmeans).to_csv(OUT/'all_connection_cell_means.csv',index=False)
for domain in ('C1','C2','K1'):
    cells=[s['id'] for s in plan['cells'] if s['condition']==domain]
    x=np.concatenate([arrays[k][0] for k in cells]);c=np.concatenate([arrays[k][1] for k in cells])
    frame=meta.copy();frame['domain']=domain;frame['n']=len(x);frame['target_mean']=x.mean(0);frame['control_mean']=c.mean(0)
    frame['target_minus_control_mean']=(x-c).mean(0);frame['status']='full_grid_not_numerically_validated'
    full_frames.append(frame)
full=pd.concat(full_frames);full.to_csv(OUT/'full_grid_connection_means_PROVISIONAL.csv',index=False)

# Primary restricted comparison: library ABA (belief/photo), indexed-list ABA (code).
# Test the other compatible cells too, as within-cell evidence, without pooling
# different template/direction mixtures into a domain comparison.
# Sign-flip max statistic uses the existing project implementation, jointly over
# 2218 targets and 2218 target-minus-control series; Bonferroni across six cells.
# Null requires joint sign symmetry about zero; not a randomized domain test.
torch.set_num_threads(1); torch.manual_seed(20261023)
rng=np.random.default_rng(20261023); draws=50000
estimate_frames=[]; boot_audit=[]
tested_cells=audit.loc[audit.compatible,'cell'].tolist()
for cid in tested_cells:
    domain=next(c['condition'] for c in plan['cells'] if c['id']==cid)
    x,c=arrays[cid]; combined=np.concatenate([x,x-c],axis=1)
    perm,threshold=permutation_test_heads(torch.tensor(combined,dtype=torch.float32),n_permutations=5000,alpha=.05/len(tested_cells))
    # Independent exhaustive check: only 2**10 sign assignments exist here.
    # Keep the project's test as primary; verify its Monte Carlo decision exactly.
    signs=2*((np.arange(1024)[:,None]>>np.arange(10))&1)-1
    exact_maxima=np.max(np.abs(signs@combined/10),axis=1)
    exact_p=np.minimum(1.,len(tested_cells)*np.mean(exact_maxima[:,None]>=np.abs(combined.mean(0))[None,:]-1e-12,axis=0))
    assert np.array_equal(perm.significant.to_numpy(),exact_p<=.05)
    mean=combined.mean(0);se=combined.std(0,ddof=1)/np.sqrt(10);valid=se>0
    # Simultaneous centered bootstrap bands, resampling prompt rows jointly.
    # Zero empirical variance is marked unestimated, not certain.
    maxima=[]
    for start in range(0,draws,1000):
        weights=rng.multinomial(10,np.full(10,.1),size=min(1000,draws-start))/10
        maxima.extend(np.max(np.abs((weights@combined[:,valid]-mean[valid])/se[valid]),axis=1))
    critical=float(np.quantile(maxima,1-.05/len(tested_cells)))
    lo=np.where(valid,mean-critical*se,np.nan);hi=np.where(valid,mean+critical*se,np.nan)
    for offset,kind in [(0,'target'),(2218,'target_minus_control')]:
        frame=meta.copy();frame['domain']=domain;frame['cell']=cid;frame['kind']=kind;frame['n']=10
        frame['mean']=mean[offset:offset+2218];frame['lower']=lo[offset:offset+2218];frame['upper']=hi[offset:offset+2218]
        frame['max_statistic_threshold']=threshold;frame['max_statistic_detected']=perm.significant.to_numpy()[offset:offset+2218]
        frame['pointwise_signflip_p_NOT_adjusted']=perm.p_value.to_numpy()[offset:offset+2218]
        frame['exact_maxstat_p_bonferroni_six_cells']=exact_p[offset:offset+2218]
        frame['positive_prompts']=(combined[:,offset:offset+2218]>0).sum(0)
        frame['negative_prompts']=(combined[:,offset:offset+2218]<0).sum(0)
        frame['control_mean']=c.mean(0)
        estimate_frames.append(frame)
    boot_audit.append(dict(domain=domain,cell=cid,bootstrap_critical=critical,signflip_threshold=float(threshold),zero_variance_columns=int((~valid).sum())))
    print('Restricted',domain,'threshold',threshold,'detected target',int(perm.significant.iloc[:2218].sum()),'detected adjusted',int(perm.significant.iloc[2218:].sum()),flush=True)
compatible_estimates=pd.concat(estimate_frames)
compatible_estimates.to_csv(OUT/'all_compatible_cell_estimates.csv',index=False)
restricted=compatible_estimates[compatible_estimates.cell.isin(primary.values())].copy()
restricted.to_csv(OUT/'restricted_connection_estimates.csv',index=False)
contrast_frames=[]
for kind in ('target','target_minus_control'):
    for left,right in [('C1','C2'),('C1','K1'),('C2','K1')]:
        a=restricted[(restricted.domain==left)&(restricted.kind==kind)].reset_index(drop=True)
        b=restricted[(restricted.domain==right)&(restricted.kind==kind)].reset_index(drop=True)
        frame=meta.copy();frame['kind']=kind;frame['left']=left;frame['right']=right
        frame['difference']=a['mean']-b['mean'];frame['lower']=a.lower-b.upper;frame['upper']=a.upper-b.lower
        frame['excludes_zero']=(frame.lower>0)|(frame.upper<0)
        contrast_frames.append(frame)
contrasts=pd.concat(contrast_frames);contrasts.to_csv(OUT/'restricted_domain_differences.csv',index=False)

# Same-route detection in each domain; never call non-detection evidence of absence.
wide=meta.copy()
for domain in primary:
    raw=restricted[(restricted.domain==domain)&(restricted.kind=='target')].reset_index(drop=True)
    adj=restricted[(restricted.domain==domain)&(restricted.kind=='target_minus_control')].reset_index(drop=True)
    for stat in ('mean','lower','upper','positive_prompts','negative_prompts'):
        wide[f'{domain}_{stat}']=raw[stat]
    wide[f'{domain}_control_mean']=raw.control_mean
    wide[f'{domain}_adjusted_mean']=adj['mean']
    wide[f'{domain}_target_detected']=raw.max_statistic_detected
    wide[f'{domain}_adjusted_detected']=adj.max_statistic_detected
    wide[f'{domain}_positive_both_detected']=(raw.max_statistic_detected & adj.max_statistic_detected & (raw['mean']>0) & (adj['mean']>0))
wide['positive_detected_domains']=wide.apply(lambda r:';'.join(d for d in primary if r[f'{d}_positive_both_detected']),axis=1)
wide['positive_raw_means_all_three']=np.all(wide[[f'{d}_mean' for d in primary]].to_numpy()>0,axis=1)
wide['positive_adjusted_means_all_three']=np.all(wide[[f'{d}_adjusted_mean' for d in primary]].to_numpy()>0,axis=1)
wide.to_csv(OUT/'all_connections_restricted.csv',index=False)
headrows=[]
for h in assign:
    sub=wide[wide.sender==h]; score=np.max(np.abs(sub[[f'{d}_mean' for d in primary]].to_numpy()),axis=1)
    best=sub.iloc[int(np.argmax(score))]
    row=dict(sender=h,**assign[h],scope=best.scope,connections_tested=len(sub),
        strongest_observed_receiver=best.receiver,strongest_observed_component=best.component,
        strongest_receiver_aliases=best.aliases,
        routes_detected_positive_all_three=int((sub.positive_detected_domains=='C1;C2;K1').sum()),
        positive_raw_means_all_three=int(sub.positive_raw_means_all_three.sum()),
        positive_adjusted_means_all_three=int(sub.positive_adjusted_means_all_three.sum()))
    for d in primary:
        row[f'{d}_strongest_route_mean']=best[f'{d}_mean'];row[f'{d}_strongest_route_control']=best[f'{d}_control_mean']
        row[f'{d}_strongest_route_adjusted']=best[f'{d}_adjusted_mean']
        row[f'{d}_positive_detected_routes']=int(sub[f'{d}_positive_both_detected'].sum())
        row[f'{d}_max_abs_raw_route_mean']=float(sub[f'{d}_mean'].abs().max())
    headrows.append(row)
heads=pd.DataFrame(headrows);heads.to_csv(OUT/'all_41_heads_restricted.csv',index=False)
summary=dict(control_rows=int(audit.control_rows.sum()),failed_controls=int(audit.failed_controls.sum()),
    nonzero_clean_answer_effects=int(audit.nonzero_answer_effects.sum()),max_clean_answer_effect=float(audit.max_abs_control_effect.max()),
    compatible_cells=audit.loc[audit.compatible,'cell'].tolist(),primary_cells=primary,connections=len(meta),
    detection_patterns=wide.positive_detected_domains.value_counts().to_dict(),
    domain_contrasts_excluding_zero=int(contrasts.excludes_zero.sum()),bootstrap=boot_audit,
    bootstrap_draws=draws,seed=20261023,permutations=5000,alpha=.05,
    source_hashes=hashes,statistical_testing_sha256=sha(ROOT/'codebase/tasks/causal_analysis/statistical_testing.py'))
(OUT/'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2),flush=True)
print('Strongest restricted target means:')
print(wide.assign(max_abs=wide[[f'{d}_mean' for d in primary]].abs().max(axis=1)).nlargest(15,'max_abs')[['sender','receiver','component','C1_mean','C2_mean','K1_mean','C1_adjusted_mean','C2_adjusted_mean','K1_adjusted_mean','positive_detected_domains']].to_string(index=False))
