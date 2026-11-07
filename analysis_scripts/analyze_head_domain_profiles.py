"""Compare saved per-head CMA effects without rerunning a language model.

Read results/aggregated/head_evidence_inventory/cma_sources.csv, verify source hashes, and
retain all 1920 heads for uncertainty calibration before displaying the frozen
41-head selection. Conditions are resolved by condition_registry, not aliases.

H0 for a comparison: the specified weighted mean intervention effects agree.
We report confidence bounds, not p-values or domain-specific-head declarations.
Within each cell, resample prompt rows jointly across heads. A standardized
bootstrap maximum absolute mean error calibrates simultaneous head bands.
Allocate alpha across ALL source cells by Bonferroni, then propagate the cell
bands through each fixed linear contrast. This does not assume row pairing or
independence across cells. Bootstrap coverage is approximate, conditional on
observed successful prompts and fixed templates, and does not certify held-out
replication. Zero observed variance yields undefined uncertainty, not certainty.
Existing statistical_testing.py zero-effect sign-flips test a different null;
they are deliberately not repurposed as cross-domain tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', '/tmp/modularsymbtom-mpl')
import numpy as np
import pandas as pd
import torch
from condition_registry import lookup, pick_sample_dir

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Cell:
    key: str
    condition: str
    template: str
    rule: str
    seed: str
    path: Path
    sha256: str
    sensitivity: bool = False
    mean: Any = None
    se: Any = None
    lower: Any = None
    upper: Any = None
    point_lower: Any = None
    point_upper: Any = None
    n: int = 0
    critical: float = float('nan')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_cell(a: np.ndarray, selected: np.ndarray, rng: np.random.Generator,
                   draws: int, alpha: float) -> dict[str, Any]:
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite [prompts, heads] scores with >=2 prompts')
    mean = a.mean(0)
    se = a.std(0, ddof=1) / np.sqrt(len(a))
    valid = se > 0
    maxima = np.empty(draws)
    picked = np.empty((draws, len(selected)))
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        w = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=stop-start)
        b = (w / len(a)) @ a
        maxima[start:stop] = np.max(np.abs((b[:, valid]-mean[valid])/se[valid]), axis=1) if valid.any() else np.nan
        picked[start:stop] = b[:, selected]
    critical = float(np.quantile(maxima, 1-alpha)) if valid.any() else float('nan')
    half = critical * se
    half[~valid] = np.nan
    pl, pu = np.quantile(picked, [.025,.975], axis=0)
    pl[~valid[selected]] = np.nan
    pu[~valid[selected]] = np.nan
    return dict(mean=mean, se=se, lower=mean-half, upper=mean+half,
                point_lower=pl, point_upper=pu, critical=critical)


def contrast_bounds(cells: dict[str, Cell], weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = {k:v for k,v in weights.items() if v != 0}
    first = cells[next(iter(weights))]
    mean = np.zeros_like(first.mean)
    lo, hi = mean.copy(), mean.copy()
    for key, weight in weights.items():
        c = cells[key]
        mean += weight*c.mean
        lo += weight*(c.lower if weight > 0 else c.upper)
        hi += weight*(c.upper if weight > 0 else c.lower)
    return mean, lo, hi


def matched_cells(cells: dict[str, Cell], left: str, right: str, sensitivity: bool=False) -> list[tuple[Cell,Cell]]:
    a = [c for c in cells.values() if c.condition==left and c.sensitivity==sensitivity]
    b = [c for c in cells.values() if c.condition==right and not c.sensitivity]
    pairs=[]
    for ca in a:
        # Controls can keep the rule whereas divergence flips it: match the base rule,
        # not the full donor/recipient rule pair. Save both actual rule labels.
        base=ca.rule.split('_')[2]
        for cb in b:
            if cb.rule.split('_')[2] != base:
                continue
            if ca.template != cb.template and 'template_indexed_list' not in (ca.template,cb.template):
                continue
            pairs.append((ca,cb))
    return sorted(pairs,key=lambda p:(p[0].template,p[0].rule,p[1].rule))


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources',type=Path,default=ROOT/'results/aggregated/head_evidence_inventory/cma_sources.csv')
    ap.add_argument('--membership',type=Path,default=ROOT/'results/aggregated/head_evidence_inventory/head_membership.csv')
    ap.add_argument('--out-dir',type=Path,default=ROOT/'results/aggregated/head_domain_comparison')
    ap.add_argument('--draws',type=int,default=50000)
    ap.add_argument('--seed',type=int,default=20261022)
    args=ap.parse_args()
    if args.draws<1000:
        ap.error('Use at least 1000 draws; production default is 50000')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    heads=pd.read_csv(args.membership)
    selected=(heads.layer.to_numpy(dtype=int)*40+heads['head'].to_numpy(dtype=int))
    assert len(np.unique(selected))==len(heads)==41
    sources=pd.read_csv(args.sources,dtype={'seed':str})
    cells:dict[str,Cell]={}
    missing_sensitivity=[]
    for r in sources.itertuples():
        cond=lookup(r.registry_id)
        assert cond.folder == r.condition, (r.registry_id,cond)
        key=f'{r.registry_id}:{r.template}:{r.rule}:canonical'
        cells[key]=Cell(key,r.registry_id,r.template,r.rule,r.seed,ROOT/r.per_prompt_path,r.sha256_per_prompt)
        if r.registry_id=='C1' and r.seed!='1':
            # Choose sample count within the exact seed-1 branch using the registry helper.
            logit=(ROOT/r.per_prompt_path).parents[2].with_name('z_seed_1_shuffle_False')/'logit'
            sample=pick_sample_dir(logit)
            if sample is not None:
                source_path=Path(sample[0])/'causal_scores_per_sample.pt'
                if source_path.exists():
                    skey=f'C1:{r.template}:{r.rule}:seed1_sensitivity'
                    cells[skey]=Cell(skey,'C1',r.template,r.rule,'1',source_path,digest(source_path),True)
                else:missing_sensitivity.append(str(source_path))
            else:missing_sensitivity.append(str(logit))
    rng=np.random.default_rng(args.seed)
    alpha=.05/len(cells)
    profiles=[];manifest=[]
    for i,c in enumerate(cells.values()):
        if digest(c.path)!=c.sha256:
            raise ValueError(f'Source hash changed: {c.path}')
        tensor=torch.load(c.path,map_location='cpu',weights_only=True)
        if not isinstance(tensor,torch.Tensor) or tuple(tensor.shape[1:])!=(48,40):
            raise ValueError(f'Unexpected tensor shape/type: {c.path}')
        a=tensor.double().numpy().reshape(len(tensor),1920)
        c.n=len(a)
        for k,v in bootstrap_cell(a,selected,rng,args.draws,alpha).items():setattr(c,k,v)
        for j,h in enumerate(heads.itertuples()):
            idx=int(selected[j])
            profiles.append(dict(cell=c.key,condition=c.condition,template=c.template,rule=c.rule,seed=c.seed,
                sensitivity=c.sensitivity,n=c.n,head_name=h.head_name,layer=h.layer,head=h.head,membership=h.membership,
                mean_cma=c.mean[idx],pointwise_bootstrap_95_lower=c.point_lower[j],pointwise_bootstrap_95_upper=c.point_upper[j],
                simultaneous_lower=c.lower[idx],simultaneous_upper=c.upper[idx],standard_error=c.se[idx]))
        manifest.append(dict(cell=c.key,path=str(c.path.relative_to(ROOT)),sha256=c.sha256,n=c.n,
            shape=list(tensor.shape),dtype=str(tensor.dtype),max_error_critical=c.critical,cell_alpha=alpha,
            zero_variance_heads=int((c.se==0).sum()),prompt_pairing_verified=False))
        print(f'[{i+1}/{len(cells)}] {c.key} n={c.n} finite=True max-error={c.critical:.3f}',flush=True)
    plans: list[dict[str, Any]]=[]
    # This is a comparison plan using canonical IDs; condition definitions remain in the registry.
    for left,right in [('C1','C2'),('C1','K1'),('C2','K1'),('C1','C5'),('C2','C6'),('K1','K5'),('Q1c','Q1a'),('Q1a','Q1b'),('K1','Q2a'),('K1','Q2b'),('Q3a','Q3b')]:
        pairs=matched_cells(cells,left,right)
        for ca,cb in pairs:
            plans.append(dict(name=f'{left}_minus_{right}:{ca.template}:{ca.rule}:{cb.rule}',left=left,right=right,
                level='cell',template=ca.template,rule=ca.rule,weights={ca.key:1.,cb.key:-1.},sensitivity=False))
        if pairs:
            weights:dict[str,float]={}
            for ca,cb in pairs:
                weights[ca.key]=weights.get(ca.key,0)+1/len(pairs)
                weights[cb.key]=weights.get(cb.key,0)-1/len(pairs)
            plans.append(dict(name=f'{left}_minus_{right}:equal_cell_mean',left=left,right=right,
                level='equal_cell_mean',template='fixed_observed_templates',rule='equal_observed_directions',weights=weights,sensitivity=False))
    for ca,cb in matched_cells(cells,'C1','C2',True):
        plans.append(dict(name=f'C1_minus_C2:seed1_sensitivity:{ca.template}:{ca.rule}',left='C1',right='C2',level='cell',template=ca.template,rule=ca.rule,weights={ca.key:1.,cb.key:-1.},sensitivity=True))
    contrasts=[];allheads=[]
    for p in plans:
        mean,lo,hi=contrast_bounds(cells,p['weights'])
        for j,h in enumerate(heads.itertuples()):
            ix=int(selected[j]);contrasts.append({k:v for k,v in p.items() if k!='weights'}|dict(head_name=h.head_name,membership=h.membership,
                mean_difference=mean[ix],simultaneous_lower=lo[ix],simultaneous_upper=hi[ix],weights_json=json.dumps(p['weights'],sort_keys=True)))
        if p['level']=='equal_cell_mean':
            for ix in range(1920):
                allheads.append(dict(comparison=p['name'],layer=ix//40,head=ix%40,mean_difference=mean[ix],simultaneous_lower=lo[ix],simultaneous_upper=hi[ix]))
    profiles_df=pd.DataFrame(profiles);cd=pd.DataFrame(contrasts)
    profiles_df.to_csv(args.out_dir/'cma_cell_profiles.csv',index=False)
    cd.to_csv(args.out_dir/'cma_head_comparisons.csv',index=False)
    pd.DataFrame(allheads).to_csv(args.out_dir/'cma_all_heads_aggregate_comparisons.csv',index=False)
    with (args.out_dir/'cma_source_manifest.json').open('w') as f:json.dump(manifest,f,indent=2)
    meta=dict(command=sys.argv,bootstrap_draws=args.draws,seed=args.seed,alpha=.05,cell_alpha=alpha,
        multiplicity_scope=f'all 1920 heads in each of {len(cells)} cells; union-bound propagation to every listed linear contrast',
        assumptions=['Prompt-row bootstrap within each cell; selected successful prompts only','Fixed templates and direction weights; not new-template inference','Unknown across-cell dependence permitted by union-bound construction','Bootstrap bands approximate; data-dependent selection not independent replication','Zero-variance uncertainty undefined'],
        missing_seed1_sensitivity=missing_sensitivity,source_manifest_sha256=digest(args.sources),membership_sha256=digest(args.membership),script_sha256=digest(Path(__file__)),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),numpy_version=np.__version__,torch_version=torch.__version__)
    (args.out_dir/'cma_analysis_config.json').write_text(json.dumps(meta,indent=2)+'\n')
    render(profiles_df,cd,heads,args.out_dir,meta)
    print(f'Wrote {len(profiles_df)} head/cell profiles and {len(cd)} head/contrast rows to {args.out_dir}',flush=True)


def render(profiles: pd.DataFrame, contrasts: pd.DataFrame, heads: pd.DataFrame, out: Path, meta: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    canonical=profiles[~profiles.sensitivity]
    ids=['C1','C2','K1','C5','C6','K5','Q1a','Q1b','Q1c','Q2a','Q2b','Q2c','Q3a','Q3b']
    order=heads.sort_values(['membership','layer','head']).head_name.tolist()
    data=canonical.groupby(['head_name','condition']).mean_cma.mean().unstack().reindex(index=order,columns=ids)
    limit=float(np.nanmax(np.abs(data.to_numpy())))
    fig,ax=plt.subplots(figsize=(12,14))
    im=ax.imshow(data.to_numpy(),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    membership=heads.set_index('head_name').membership.to_dict()
    ax.set_yticks(range(len(order)),[f'{h}  [{membership[h]}]' for h in order],fontsize=8)
    ax.set_xticks(range(len(ids)),ids,rotation=45,ha='right')
    ax.set_title('Saved individual-head CMA effects: equal cell means\nRaw logit-margin change; domain/template and filtered-population confounds remain',fontsize=11)
    fig.colorbar(im,ax=ax,label='CMA: patched − clean target-versus-original logit margin (unbounded)')
    fig.tight_layout();fig.savefig(out/'cma_profiles.png',dpi=170);plt.close(fig)
    report=['# Existing-data head comparison\n',
      'This analysis ran on saved tensors; no language-model inference was performed.\n',
      'Metric: CMA is the patched minus clean target-versus-original answer logit margin. Zero means no change, positive means movement toward the specified counterfactual answer, and the range is unbounded. It is not accuracy, probability, or a head-ablation effect. Cell means are equally weighted in summaries; templates and directions remain separately available.\n',
      'Uncertainty: joint prompt-row bootstrap within each cell, calibrated using the maximum standardized mean error across all 1,920 heads. Per-cell error budgets sum to 0.05 across all source cells. Propagated bands therefore target simultaneous 95% coverage across the displayed linear contrasts under the bootstrap approximation, without assuming independence or pairing across cells. This is conservative; historical head selection is not an independent replication. Zero-variance uncertainties are left blank.\n',
      'All cells are conditioned on both donor and recipient passing original accuracy filters. Cross-domain differences are differences in intervention effects on these filtered prompt populations, not an isolated causal effect of domain. Code has only one template.\n',
      '## Strictly belief-only heads\n',
      'Each entry is mean difference [simultaneous lower, upper], in raw logit-margin units. Positive belief−photo/code means a larger measured belief patch effect; a bound covering zero is unresolved, not evidence of equivalence. These averages are conditional on the observed templates.\n',
      '| Head | Belief − photo | Belief − code |\n|---|---|---|\n']
    selected=heads[heads.membership=='belief'].head_name
    for h in selected:
        fields=[]
        for label in ['C1_minus_C2:equal_cell_mean','C1_minus_K1:equal_cell_mean']:
            r=contrasts[(contrasts.name==label)&(contrasts.head_name==h)].iloc[0]
            fields.append(f'{r.mean_difference:.3f} [{r.simultaneous_lower:.3f}, {r.simultaneous_upper:.3f}]')
        report.append(f'| {h} | '+ ' | '.join(fields)+' |\n')
    report += ['\n## Files\n',
      '- `cma_cell_profiles.csv`: every selected head, condition/template/direction, both pointwise 95% bootstrap intervals and family-wide cell bands. Do not confuse their multiplicity scopes.\n',
      '- `cma_head_comparisons.csv`: all planned cell and aggregate differences, exact weights, uncertainty, seed-1 sensitivity.\n',
      '- `cma_all_heads_aggregate_comparisons.csv`: aggregate contrasts for all 1,920 heads, before selection.\n',
      '- `cma_source_manifest.json` and `cma_analysis_config.json`: hashes, shapes, seeds, method and source revision.\n',
      '- `cma_profiles.png`: exploratory raw-score profile, not a significance map.\n',
      '\nControls answer different intervention questions: C1−C5 and the Q contrasts do not subtract out a uniquely identified semantic component. They constrain hypotheses but do not establish what a head exclusively encodes.\n',
      f'\nSeed-1 sensitivity missing sources: {len(meta["missing_seed1_sensitivity"])}. No matching row IDs are assumed across source cells.\n']
    report_text=''.join(report)
    for beginning in ['Metric:', 'Uncertainty:', 'All cells', '| Head |']:
        report_text=report_text.replace('\n'+beginning, '\n\n'+beginning)
    for heading in ['# Existing-data head comparison', '## Strictly belief-only heads', '## Files']:
        report_text=report_text.replace(heading+'\n', heading+'\n\n')
    (out/'CMA_REPORT.md').write_text(report_text)

if __name__=='__main__':main()
