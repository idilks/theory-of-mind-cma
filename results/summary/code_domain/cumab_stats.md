### How many heads does it take? (k½)

First k at which accuracy drops below 50% — ablation curve vs the same-layer control curve. A small ablation k½ next to a large control k½ = the damage is specific to those heads, not to removing heads per se.


**quote-prefill (primary)**

| eval | C1 ablation | C1 control | C5 ablation | C5 control |
|---|---|---|---|---|
| code copy (false-belief equiv) | 6 | 29 | 9 | 160 |
| code alias (true-belief equiv) | 33 | 78 | 69 | 192 |
| ToM false belief | 8 | 31 | 52 | 512 |
| ToM true belief | 29 | 432 | 82 | 75 |
| ToM basic retrieval | 38 | 400 | 63 | 496 |

**free generation**

| eval | C1 ablation | C1 control | C5 ablation | C5 control |
|---|---|---|---|---|
| code copy (false-belief equiv) | 8 | 113 | 41 | 368 |
| code alias (true-belief equiv) | 192 | 128 | 496 | 288 |
| ToM false belief | 3 | 19 | 13 | 512 |
| ToM true belief | 63 | 208 | 160 | 160 |
| ToM basic retrieval | 63 | 400 | 160 | 480 |
