"""Render the verified repaired-study report and complete 41-head figure."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/path-h200-analysis-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
h=pd.read_csv(P/'all_41_heads.csv');w=pd.read_csv(P/'all_connections.csv');a=pd.read_csv(P/'control_audit_by_cell.csv');e=pd.read_csv(P/'cell_estimates.csv');d=pd.read_csv(P/'domain_estimates.csv');stats=json.loads((P/'statistics.json').read_text())
order=['belief','photo','code','belief + photo','belief + code','belief + photo + code']
short={'belief':'B only','photo':'P only','code':'K only','belief + photo':'B+P','belief + code':'B+K','belief + photo + code':'B+P+K'}
h['_order']=h.membership.map(dict(zip(order,range(len(order)))))
h=h.sort_values('_order',kind='stable').reset_index(drop=True)
labels=[f'{r.sender} → {r.strongest_observed_receiver} {r.strongest_observed_component.upper()} [{short[r.membership]}]' for r in h.itertuples()]
fig,axes=plt.subplots(1,2,figsize=(13,17),sharey=True,gridspec_kw={'wspace':.12})
limit=.27
for ax,suffix,title in zip(axes,['mean','adjusted_mean'],['Selected sender: mean path effect','Selected sender minus matched control']):
    values=h[[f'{x}_{suffix}' for x in ('C1','C2','K1')]].to_numpy()
    ax.imshow(values,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(3),['Belief','Photo','Code']);ax.xaxis.tick_top();ax.set_title(title,pad=30,fontsize=11)
    ax.set_yticks(range(41),labels,fontsize=8.5)
    for i in range(41):
        for j in range(3):ax.text(j,i,f'{values[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(values[i,j])>.17 else '#111111')
    for i in range(1,41):
        if h.iloc[i].membership!=h.iloc[i-1].membership:ax.axhline(i-.5,color='#333333',linewidth=1.5)
    ax.spines[['top','right','bottom','left']].set_visible(False)
fig.suptitle('All 41 senders — complete H200-validated grid',fontsize=16,y=.985)
fig.text(.5,.958,'Both directions; three templates each for belief/photo, one for code. Signed logit-margin change.',ha='center',fontsize=10)
fig.text(.5,.035,'One descriptive route per head: largest average absolute domain mean. Same-data selection can exaggerate effects.\nValues are means, not significance labels. No direct domain contrast excludes zero under the simultaneous bounds.\nB/P/K labels describe original head selection; KV denotes a native key/value group shared by several query heads.',ha='center',fontsize=9)
fig.subplots_adjust(left=.32,right=.97,top=.91,bottom=.08)
fig.savefig(P/'all_41_connections.png',dpi=160);plt.close(fig)

lines=['# H200 repairs: complete 41-head connection analysis','',
'Updated October 23, 2026. **All eight repair arms pass; all 14 combined cells are now numerically compatible.** This report supersedes the six-cell clean-control analysis for the full-grid connection comparison. Original and repaired raw studies are preserved separately.','',
'## Repair validation','',
'- All eight repairs have 20 scope/prompt checkpoints each and record NVIDIA H200 NVL on `h200b`, with the original Torch 2.5.1+cu121, TransformerLens 2.16.1, and Transformers 4.53.2 versions. Snapshot, prompt, task and source hashes pass. Per-task Slurm job IDs can differ from the parent array ID and are retained in `repair_runtime.json`.',
'- The four rerun clean-control cells contribute 4,400 controls, all with exactly zero full-logit error. Together with retained H200 controls, all 15,400 clean transplants pass, with zero error in the checked diagnostics.',
'- For the four repaired path cells, retained controls still truthfully reference the old path-file hashes. The analysis verifies those old references, then matches exact prompt, sender, receiver component, route aliases, batch width and batch row to the repaired measurements. Selected path and control single/batch margins match exactly in all 14 cells.',
'- All 621,040 selected/control-sender path rows pass the saved identity diagnostics. No selected measurement has zero receiver change with a nonzero path effect. This removes the observed numerical failure; it is not a proof about unsaved internal activations.',
'- All 918 source files hashed by the audit were independently rechecked after analysis. Original raw outputs were not edited.','',
'| Repaired arm | Cell | Clean controls | Failed | Baseline mismatches | Status |',
'|---|---|---:|---:|---:|---|']
for r in a[a.path_repaired|a.control_repaired].itertuples():
    lines.append(f"| {'Path' if r.path_repaired else 'Clean control'} | {r.cell} | {r.control_rows} | {r.failed_controls} | {r.single_margin_mismatches+r.batch_margin_mismatches} | PASS |")
lines += ['','The repaired path values genuinely replace the affected original values; they are not assumed to be equal. Per-prompt changes are saved in `old_vs_repaired_path_changes.csv`.','',
'## What is measured and compared','',
'**Path effect** is the transplanted-run answer margin, `logit(causal answer) − logit(original answer)`, minus the same margin in the untouched batch reference row. It is signed, unbounded and measured in logit units. Positive means movement toward the designated causal answer; it is not an accuracy percentage. **Target minus matched control** subtracts the effect of a fixed unselected sender from the same layer, acting on the identical receiver/component and prompt. That comparison is against one particular control head, not a population of random heads. The zero clean-transplant diagnostic is a different control; its value is never subtracted as a numerical correction.',
'',
'The 41 selected divergence heads contribute **2,218 distinct connections**: 39 senders × 54 receiver components, plus two layer-46 senders × 56 exploratory layer-47 components. Query (Q) routes are per query head; key (K) and value (V) routes address native shared groups, counted once with aliases preserved. Each selected route has a same-layer control sender. The assay measures final-token direct routes with intermediate attention/MLP writes frozen; it does not cover all possible receivers, other token positions, or indirect routes through changing intermediate writers.',
'',
'Each of 14 cells has 10 saved prompts. Full-grid means give equal weight to the six belief cells, six photo cells and two code cells within each domain. Belief/photo use the same three template labels and two transplant directions; code uses indexed-list in both directions. These are **fixed tested-task averages**, not general population effects. Direction-specific and each-story-template-versus-code tables are also supplied. Shared generation seeds and semantic content can induce cross-cell dependence; row numbers are not asserted to be matched statistical pairs.',
'',
'## Inference and limits','',
'1. **Within-cell tests:** null mean path effect = 0, or mean selected-minus-control effect = 0. Enumerate all 1,024 prompt-level sign assignments. The maximum absolute mean across 2,218 raw and 2,218 control-adjusted series controls the route/comparison search within a cell; multiply by 14 for all cells, at overall alpha 0.05. Exactness requires joint sign symmetry under the null, which is an assumption, not a randomized treatment assignment. All positive and negative effects enter the test.',
'2. **Domain contrasts:** null difference between the specified fixed-domain averages = 0. Jointly resample the 10 prompt rows within each cell 50,000 times. Maximum standardized centered-bootstrap errors provide simultaneous bounds across all 4,436 columns, with alpha allocated across 14 cells. Average cell endpoints for domain bounds, then subtract endpoints for contrasts. This remains conservative without assuming cross-cell independence or prompt pairing. The bootstrap coverage is approximate with n=10, not an exact small-sample guarantee. These interval calculations are not inversions of the unstudentized sign test, so their decisions can differ.',
'3. **Zero observed variance:** an empirical constant is assigned unestimated bounds, not certainty. Missing bounds propagate into averages and contrasts. In the full-grid comparison, 9,143 of 13,308 raw/control-adjusted contrasts have finite bounds; 4,165 are unestimated. No finite contrast excludes zero. The same conclusion holds for the direction and template breakdowns.',
'4. A **shared positive connection** requires positive support in all three domains; positive means alone are descriptive. **Domain preference** requires a direct domain contrast, not significance in one domain and non-significance in another. Even a significant preference would not prove exclusive use in one domain. These data establish neither a shared-route count nor an exclusive-route count.',
'',
'## Findings','',
'**L39H23 → L43H30 Q is the clearest remaining candidate, with a corrected selected-versus-control effect in code ABB.** The selected effect is +0.3875, the matched control is −0.0625, and their difference is +0.4500; all 10 prompt differences are positive. Exact full-study corrected p = 0.02734375. The raw effect alone does not pass that exact test (p = 0.1640625). The numerically invalid code control has now been repaired, so this comparison is usable.',
'',
'Averaging both code directions gives selected effect +0.24375 and selected-minus-control +0.26250. The latter has a borderline positive approximate simultaneous interval [+0.00152, +0.52348]; the raw interval includes zero. Neither direct code-minus-belief nor code-minus-photo contrast excludes zero. Thus this supports a **code-cell contribution relative to its matched control**, not a code-specific mechanism.',
'',
'**The earlier photo candidate, L39H23 → L45H38 Q, remains descriptively stronger in photo.** Its library-book ABA effect is unchanged at +0.2000, with control 0. The old six-cell correction gave p = 0.046875; testing the complete 14-cell family gives p = 0.109375. This change comes from the larger multiple-testing family, not a reversal of the measurement. Its approximate bootstrap interval in that cell is positive, while the primary exact sign test does not pass. Photo effects also vary by template/direction.',
'',
'Full-grid means below are **selected effect (selected minus matched control)**. Each row is the same sender/receiver/component across all domains; positive values do not certify sharing or preference.','',
'| Connection | Belief | Photo | Code |','|---|---:|---:|---:|']
for receiver in ('L43H30','L45H38'):
    r=w[(w.sender=='L39H23')&(w.receiver==receiver)&(w.component=='q')].iloc[0]
    vals=[f'{r[x+"_mean"]:+.4f} ({r[x+"_adjusted_mean"]:+.4f})' for x in ('C1','C2','K1')]
    lines.append(f'| L39H23 → {receiver} Q | '+ ' | '.join(vals)+' |')
lines += ['','**No corrected route detection from any of the 32 non-shared sender heads.** All 12 belief-only, seven photo-only, two code-only, ten belief+photo, and one belief+code sender are included. None has a connection passing the full-set within-cell exact test; neither do eight of the nine shared senders. This does not contradict their whole-head causal effects: this narrower assay measures a specific direct route, and small effects, quantized outputs, ten prompts/cell, and broad multiplicity limit resolution. It does not establish that beliefs are processed identically or differently.',
'',
'| Original selection membership | Heads | Connections | Connections with any corrected cell detection |','|---|---:|---:|---:|']
for group in order:
    g=h[h.membership==group]
    lines.append(f'| {group} | {len(g)} | {g.connections_tested.sum()} | {g.connections_detected_any_cell.sum()} |')
lines += ['',
'**Shared-versus-specific ratio: still not established.** No connection has supported positive effects in all three domains under both raw and matched-control comparisons, and no direct domain contrast is resolved. This is insufficient evidence, not a ratio of zero shared to zero specific. Descriptively, 1,275/2,218 routes (57.5%) have positive raw means in all three domains, falling to 283/2,218 (12.8%) after matched-control subtraction. These sign counts must not be presented as established shared circuits. Original head-selection membership (9 shared, 32 non-shared) is a separate quantity.',
'',
'## Every sender','',
'The table and figure show one descriptive route per head, selected by the largest average absolute domain mean, weighting the three domains equally. Selecting on these data exaggerates apparent extrema; this is not an inferential selection of a best route. Complete unfiltered results for all routes, controls, signs and native-group aliases are in `all_connections.csv`, `cell_estimates.csv`, and `domain_estimates.csv`.','',
'| Sender | Original membership | Descriptive route | Belief | Photo | Code |','|---|---|---|---:|---:|---:|']
for r in h.itertuples():
    lines.append(f'| {r.sender} | {r.membership} | {r.strongest_observed_receiver} {r.strongest_observed_component.upper()} | {r.C1_mean:+.4f} | {r.C2_mean:+.4f} | {r.K1_mean:+.4f} |')
lines += ['','![All 41 senders](all_41_connections.png)','',
'## Files and reproduction','',
'- [All 41 heads](all_41_heads.csv) and [all 2,218 connections](all_connections.csv).',
'- [All cell estimates and exact tests](cell_estimates.csv); [fixed-grid domain estimates](domain_estimates.csv); [direct domain contrasts](domain_differences.csv).',
'- [Repair/retained-cell audit](control_audit_by_cell.csv), [individual clean controls](clean_controls.csv), [source hashes](audit.json), [statistical settings](statistics.json).',
'- [Validated source selection](validated_sources.json) identifies exactly which original and repaired roots were combined.',
'',
'Run from the repository root in the activated project environment:',
'```bash',
'python results/aggregated/path_patching_h200_analysis/audit_repairs.py',
'python results/aggregated/path_patching_h200_analysis/analyze.py',
'python results/aggregated/path_patching_h200_analysis/verify_analysis.py',
'python results/aggregated/path_patching_h200_analysis/render_report.py',
'```',
'These scripts read saved experiment outputs and write only analysis artifacts. No new GPU experiment was launched.','']
(P/'REPORT.md').write_text('\n'.join(lines))
print('Wrote REPORT.md and all_41_connections.png')
