# What limits the repaired path study, and what mechanism is supported?

October 23, 2026. Existing saved data only; no experimental changes or new GPU runs.

**The limitation is not a choice between small effects and ten prompts. Most routes are weak relative to their variability, while the leading candidates are also constrained by the discreteness of the test, multiple comparisons, and conservative domain intervals.** The previous non-detections cannot be interpreted as evidence that the mechanism is absent.

## Readout and scope

Path effect is a signed, unbounded change in the causal-versus-original answer logit margin. Positive favors the causal answer. Selected-minus-control subtracts the same-route intervention through a fixed same-layer unselected head. Standard deviation describes prompt-to-prompt variation; standard error describes uncertainty in the mean within that cell (standard deviation divided by sqrt(10)). Zero clean-transplant error establishes numerical consistency of the controls, not infinite precision of the answer readout.

There are 2,218 routes × 14 cells = 31,052 route–cell tests for each of the two comparisons. Counts below refer to route–cell tests, not unique connections or independent samples. Prompt rows, not receivers, are the sample units.

## 1. An exact sample-size/testing constraint

For a two-sided sign-flip test with m nonzero observations, even the smallest possible p-value is at least 2 / 2^m. Zero observations duplicate sign assignments without adding information. Multiplying by 14 cells gives these floors:

| Nonzero effects among 10 prompts | Smallest possible corrected p |
|---|---:|
| 8 | 0.109375 |
| 9 | 0.0546875 |
| 10 | 0.02734375 |

The route maximum-statistic correction can only increase these values. With n=10, any zero effect therefore prevents passing alpha=0.05 under the current test. A mixture of positive and negative effects also prevents passing: besides the observed sign pattern and its inverse, aligning the signs creates at least two more assignments with equal or larger absolute mean. Thus all ten nonzero effects must have the same sign even to be eligible. This is a necessary condition, not sufficient after the maximum over other routes.

The result is severe: 31,037/31,052 raw route–cell tests already cannot pass because of zeros; the remaining 15 have mixed signs. None is eligible. For selected-minus-control, 31,027 cannot pass because of zeros; of the remaining 25, only one has all ten effects in the same direction—the detected code ABB connection.

The photo candidate L39H23 → L45H38 Q has mean +0.200, standard error 0.0382, eight positive effects, two zeros and no negative effects. Its pointwise and within-cell maximum-statistic p are both 0.0078125; the 14-cell correction makes p=0.109375, exactly its floor. Calling this simply a small effect would be misleading. Leaving out any one prompt leaves its mean between +0.1806 and +0.2222.

The raw code ABB L39H23 → L43H30 Q effect has mean +0.3875, nine positive effects and one zero. Its pointwise p=0.00390625; within-cell corrected p=0.01171875; full-study p=0.1640625. Selected-minus-control has ten positive effects, mean +0.4500, and full-study p=0.02734375. Its leave-one-prompt-out means remain +0.4028 to +0.4722. These stability checks are descriptive, not independent confirmation.

## 2. Most effects are weak relative to their variability; output precision may also matter

Across the 31,052 raw route–cell summaries, median absolute mean = 0.025, median standard deviation = 0.1054, median standard error = 0.0333 logit units. These are separate medians, not a standardized effect-size estimate. About 67.1% of raw prompt-level path effects are exactly zero (67.5% after matched-control subtraction).

The smallest observed nonzero raw effect in a cell ranges from 0.03125 to 0.25; medians of nonzero magnitudes range from 0.125 to 0.25. The model ran in bfloat16, and the scoring code converts output logits to float32 only after the forward pass. The observed coarse values are consistent with limited readout precision. The saved scalar margins cannot tell us which zeros reflect genuinely ineffective interventions versus changes lost to numerical precision. Passing H200 clean controls removes the observed batching artifact; it does not resolve this precision question.

Removing multiplicity does not reveal a broad strong signal:

| Comparison | Pointwise p ≤ .05 | After maximum over routes/comparisons within cell | After additional 14-cell correction |
|---|---:|---:|---:|
| Raw selected route | 118 | 3 | 0 |
| Selected minus matched control | 172 | 4 | 1 |

Pointwise counts are diagnostics only. They are not discoveries; thousands of tests can produce nominal positives under nulls. Thresholds here come from the same exact sign assignments, so the table isolates the cost of the two correction stages without switching statistical tests.

## 3. The previous domain comparison is deliberately conservative

The earlier analysis constructs simultaneous intervals within each of 14 cells, then averages their endpoints and subtracts endpoints across domains. The cell half-widths are about 4.0–4.50 standard errors. Averaging endpoints preserves a conservative bound without assuming cross-cell independence, but it does not gain the usual precision from pooling independent samples. Unestimated bounds from zero empirical variance also propagate through the averages. Therefore “no resolved domain contrast” is conditional on that analysis, not a precise negative finding.

For L39H23 → L43H30 Q, the full-grid code-minus-belief selected-minus-control difference is +0.2031. The previous simultaneous interval is [−0.2652, +0.6715], with half-width 0.4684. **If** all contributing cells were independent, its standard error would be 0.0486; the reported half-width is 9.63 times that standard error. This comparison diagnoses conservatism; it is not a new confidence interval or significance claim. For code-minus-photo, the difference is +0.2688 and the analogous factor is 10.27.

The generator resets the same random seed for every cell, and some prompt content repeats across conditions/directions; saved pair IDs are cell-specific. Independence and semantic pairing must not be assumed merely to shrink the intervals. The next analysis priority is to audit which stimuli can be paired or grouped, then calibrate domain contrasts directly with the supported dependence structure. This changes no experiment and requires no new GPU data. Do not relax thresholds after observing the candidates and call that confirmation.

## 4. What causal mechanism do we have?

**Strong evidence for causal influence at a broader scale exists; a complete sender-to-reader explanation does not.**

- Whole-head transplants in the validated dataset strongly change the answer margin across domains. For example, L35H30 gives mean changes +4.711 in belief, +6.429 in photo, +2.175 in code; matched-control-subtracted values are +4.686, +6.416, +2.181. L29H31 gives +5.086, +1.131, +1.800. These are interventions on heads, so they establish causal influence on these measured prompts. They do not identify which downstream path carries that influence or what abstract variable the head represents.
- Earlier residual-subspace steering causally changes answers with direction controls. In the saved belief run (225 true-belief prompts), rank-one injection at coefficient 2 changes the mean alternative-answer probability from 0.0723 to 0.1715; norm-matched random and shuffled-label controls give 0.0931 and 0.1328. This score is p(alternative)/(p(alternative)+p(correct)), ranges 0–1, and is not the fraction of prompts whose answer flips. These are existing summary values, not a new statistical reanalysis of that study.
- The recovered geometric screen finds preferential query projection of the fitted residual direction; it is evidence about alignment, not causal mediation through queries. The steering intervention changes the residual stream broadly. Those observations support the hypothesis that condition-related residual information helps control retrieval, but do not prove a complete particular-head → residual direction → reader query → answer chain. They also do not isolate veracity from location/index content.
- The new path experiment adds a localized causal result: L39H23 → L43H30 Q in code ABB exceeds its matched sender control. Domain specificity is not required for that result to be useful. It does not demonstrate that queries dominate all key/value routes, or that this route explains most of the whole-head effect.

As an additional **exploratory aggregate diagnostic**, averaging the separate path interventions across all 39 main-study senders and their receiver routes, then giving domains equal weight, gives selected mean +0.0160 versus control +0.0149; difference +0.00107, approximate simultaneous interval [−0.00489,+0.00702]. The selected-minus-control query-versus-value average is +0.00049, interval [−0.00851,+0.00949]. No aggregate cell test survives its 13-quantity/14-cell correction. These are averages of distinct interventions, NOT a joint head-group intervention, total causal effect, or recovery fraction. Opposing effects and inactive routes can dilute a sparse circuit, so this does not exclude a distributed mechanism.

## Decision before more runs

Do not prescribe a larger sample size from these ten prompts or treat bootstrap resampling them as new information. First resolve the existing comparison's dependence/aggregation assumptions and the readout-resolution limitation. If a targeted confirmation is then warranted, specify its routes, contrasts and testing family before using new prompts; reducing the discovery family retrospectively would not provide independent confirmation. More prompts could improve detection of the two candidates, but these data cannot establish how many would suffice or whether most other routes have meaningful effects. No new run is currently recommended by this diagnostic alone.

## Reproduction and evidence

- Run `python results/aggregated/path_patching_h200_analysis/diagnostics/check_limits.py` in the activated project environment.
- `summary.json`, `detection_limits_by_cell.csv`, `all_route_diagnostics.csv`: effect distributions and exact correction stages.
- `candidate_diagnostics.csv`: every cell for both earlier candidates, including all ten values and leave-one-out summaries.
- `observed_resolution.csv`: observed nonzero increments, without asserting they are universal numerical resolution limits.
- `domain_interval_conservatism.csv`: illustrative independent-cell standard errors versus the previous simultaneous bounds; independence is not certified.
- `whole_head_effects_validated.csv`, `whole_head_means.csv`: whole-head effects extracted once per sender/prompt from the accepted source arms, verifying equality across repeated receiver rows.
- `global_separate_intervention_averages.csv`, `global_domain_averages.csv`: aggregate diagnostic with main-study/exploratory scope kept separate.
- `existing_steering_example.csv`: excerpt from `results/circuit/stage2_steering/Qwen2.5-14B-Instruct/L41_main/summary.csv`; recovered-method provenance is in `docs/path_patching/README.md`.
