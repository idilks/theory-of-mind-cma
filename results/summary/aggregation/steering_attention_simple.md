# Attention before and after steering

Mean attention from the final prompt token to the correct-answer and alternative locations, averaged over 26 answer-related heads in layers 42–46 of Qwen2.5-14B-Instruct. The correct answer is fixed by the original story: the old location for false belief and the current location for true belief. Location mass sums all token positions overlapping any occurrence of that location's name. Heads are averaged within each prompt before averaging across the 225 held-out prompts per condition. Error bars are descriptive 95% bootstrap confidence intervals over prompts (5,000 resamples), copied unchanged from the full figure's saved table. No significance stars or new hypothesis tests are added.

Steering adds (injection) or subtracts (deletion) the rank-1 direction at the final token immediately before layer 42, with coefficient α = 4. The heads themselves are not directly edited. Both panels share an axis. Percentages retain the total attention denominator; omitted tokens carry the remaining mass.

The clean mean is higher for the correct than the alternative location in both conditions. Deletion on false belief lowers correct-location attention and raises alternative-location attention. Injection on true belief lowers attention to both locations, with a larger decline at the correct location. In both cases the ordering reverses. This plot alone does not establish that attention changes mediate the answer change or that every head behaves this way.

## Limit of the available evidence

The steering attention.csv files save only mass_believed and mass_current per head and prompt. The runner discards full attention patterns after summing these masks. No token-level steering attention capture was found in the local results/circuit or results/circuit trees. The older lookback captures use different prompt/head selections and cannot establish a ranking for this experiment. Consequently we cannot claim that the correct-answer location is attended more than every other token or location. That claim needs matching full token-level attention measurements and a defined comparison between multi-token location spans and individual tokens.

Source: steering_attention_bars.csv in this directory. Full appendix figure: steering_attention_bars.png / .pdf. Regenerate this figure with `python analysis_scripts/circuit/render_steering_attention_simple.py`.
