# Ablation baseline comparison


## false_belief flip_score by run and k

| run | k=0 | k=11 | k=13 (max) | k=19 (max) | k=21 | k=26 (max) | k=32 (max) | k=7 (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decomp belief_only | 0.010 | 0.014 | 0.015 |  |  |  |  |  |
| decomp intersection (B rank) | 0.010 | 0.459 |  | 0.402 |  |  |  |  |
| decomp intersection (P rank) | 0.010 | 0.143 |  | 0.406 |  |  |  |  |
| decomp photo_only | 0.010 |  |  |  |  |  |  | 0.010 |
| matched_sys / no_prefill | 0.014 | 0.863 |  |  | 0.742 |  | 0.702 |  |
| matched_sys / prefill | 0.010 | 0.420 |  |  | 0.337 |  | 0.487 |  |
| neutral_sys / prefill | 0.256 | 0.636 |  |  | 0.643 |  | 0.615 |  |
| photo_heads / sys+prefill | 0.010 | 0.143 |  |  | 0.293 | 0.275 |  |  |


## k=0 baseline flip_score (φ) across eval_variants — the 0.37 mystery

| run | basic_ab | basic_ba | false_belief | false_photo | true_belief | true_photo |
| --- | --- | --- | --- | --- | --- | --- |
| decomp belief_only | 0.118 | 0.035 | 0.010 | 0.010 | 0.092 | 0.010 |
| decomp intersection (B rank) | 0.115 | 0.034 | 0.010 | 0.010 | 0.089 | 0.010 |
| decomp intersection (P rank) | 0.118 | 0.035 | 0.010 | 0.010 | 0.092 | 0.010 |
| decomp photo_only | 0.115 | 0.034 | 0.010 | 0.010 | 0.089 | 0.010 |
| matched_sys / no_prefill | 0.052 | 0.027 | 0.014 | 0.010 | 0.024 | 0.010 |
| matched_sys / prefill | 0.118 | 0.035 | 0.010 | 0.010 | 0.092 | 0.010 |
| neutral_sys / prefill | 0.013 | 0.011 | 0.256 | 0.010 | 0.029 | 0.010 |
| photo_heads / sys+prefill | 0.118 | 0.035 | 0.010 | 0.010 | 0.092 | 0.010 |


## false_belief flip_rate by run and k

| run | k=0 | k=11 | k=13 (max) | k=19 (max) | k=21 | k=26 (max) | k=32 (max) | k=7 (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decomp belief_only | 0.000 | 0.000 | 0.000 |  |  |  |  |  |
| decomp intersection (B rank) | 0.000 | 0.333 |  | 0.271 |  |  |  |  |
| decomp intersection (P rank) | 0.000 | 0.104 |  | 0.229 |  |  |  |  |
| decomp photo_only | 0.000 |  |  |  |  |  |  | 0.000 |
| matched_sys / no_prefill | 0.000 | 0.875 |  |  | 0.792 |  | 0.708 |  |
| matched_sys / prefill | 0.000 | 0.312 |  |  | 0.292 |  | 0.333 |  |
| neutral_sys / prefill | 0.208 | 0.625 |  |  | 0.667 |  | 0.562 |  |
| photo_heads / sys+prefill | 0.000 | 0.104 |  |  | 0.188 | 0.104 |  |  |


## false_belief mean_p_correct by run and k

| run | k=0 | k=11 | k=13 (max) | k=19 (max) | k=21 | k=26 (max) | k=32 (max) | k=7 (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decomp belief_only | 0.986 | 0.983 | 0.986 |  |  |  |  |  |
| decomp intersection (B rank) | 0.988 | 0.474 |  | 0.539 |  |  |  |  |
| decomp intersection (P rank) | 0.986 | 0.800 |  | 0.535 |  |  |  |  |
| decomp photo_only | 0.988 |  |  |  |  |  |  | 0.975 |
| matched_sys / no_prefill | 0.667 | 0.125 |  |  | 0.239 |  | 0.229 |  |
| matched_sys / prefill | 0.986 | 0.533 |  |  | 0.646 |  | 0.428 |  |
| neutral_sys / prefill | 0.726 | 0.363 |  |  | 0.350 |  | 0.316 |  |
| photo_heads / sys+prefill | 0.986 | 0.800 |  |  | 0.613 | 0.545 |  |  |


## Random-null references (false_belief)

### null: neutral_sys / no_prefill (21-head target)

| k | metric | target | null_mean | null_std | null_p05 | null_p95 | p_target_geq_null |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | flip_rate | 0.667 | 0.389 | 0.042 | 0.354 | 0.439 | 0.000 |
| 11 | flip_score | 0.638 | 0.415 | 0.043 | 0.376 | 0.454 | 0.000 |
| 11 | mean_p_correct | 0.119 | 0.293 | 0.031 | 0.256 | 0.319 | 1.000 |
| 21 | flip_rate | 0.625 | 0.389 | 0.058 | 0.333 | 0.461 | 0.010 |
| 21 | flip_score | 0.636 | 0.417 | 0.054 | 0.368 | 0.484 | 0.000 |
| 21 | mean_p_correct | 0.133 | 0.290 | 0.041 | 0.238 | 0.324 | 1.000 |

### null: matched_sys / prefill (32-head target)

| k | metric | target | null_mean | null_std | null_p05 | null_p95 | p_target_geq_null |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | flip_rate | 0.312 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 |
| 11 | flip_score | 0.397 | 0.007 | 0.006 | 0.000 | 0.019 | 0.000 |
| 11 | mean_p_correct | 0.543 | 0.943 | 0.009 | 0.928 | 0.954 | 1.000 |
| 21 | flip_rate | 0.417 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 |
| 21 | flip_score | 0.401 | 0.007 | 0.006 | 0.000 | 0.017 | 0.000 |
| 21 | mean_p_correct | 0.581 | 0.937 | 0.016 | 0.907 | 0.955 | 1.000 |
