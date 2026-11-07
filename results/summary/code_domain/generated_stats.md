### Cell inventory

| condition | cells | seeds | n per cell | sig heads (per cell) |
|---|---|---|---|---|
| code abstract | 2 | 0 | 148,148 | 5,13 |
| code answer-changes | 2 | 0 | 149,148 | 17,20 |
| code control | 2 | 0 | 148,150 | 9,14 |
| C1 belief | 6 | combined | 188,194,152,146,198,198 | 37,37,13,26,31,19 |
| C2 photo | 6 | 1 | 149,150,144,139,150,150 | 16,3,20,25,14,1 |
| C3 ans-changes belief | 6 | combined | 188,188,185,142,198,198 | 51,33,34,42,38,26 |
| C4 ans-changes photo | 6 | 1 | 150,150,139,139,150,150 | 9,8,9,8,13,7 |
| C5 loc-only | 6 | 1 | 142,142,147,147,150,150 | 18,16,25,26,15,16 |
| C6 loc-only photo | 6 | 1 | 150,150,150,150,150,150 | 10,9,8,8,10,10 |
| C7 state-only | 6 | 1 | 149,150,150,150,130,129 | 9,8,12,10,15,15 |

### Headline correlations (fingerprints over 1920 heads)

| pair | r | R² | permutation p (10k label shuffles) |
|---|---|---|---|
| code abstract ↔ C1 belief | 0.867 | 0.752 | <1e-4 |
| code abstract ↔ C2 photo | 0.851 | 0.724 | <1e-4 |
| code control ↔ C5 loc-only | 0.706 | 0.499 | <1e-4 |
| code control ↔ C7 state-only | 0.662 | 0.438 | <1e-4 |
| code abstract ↔ code control | 0.067 | 0.005 | 0.0199 |

### Direction consistency within code conditions

| condition | r(direction 1, direction 2) |
|---|---|
| code abstract | 0.664 |
| code answer-changes | 0.932 |
| code control | 0.823 |

### H2-analog decomposition

r(code_ac, code_abs + code_ctl) = **0.995** vs abs-only 0.461, ctl-only 0.910
— ToM benchmark on the same sources: r(C3, C1 + C5) = 0.980.

![](code_domain/h2_decomp_code.png){width=65%}
