# Steering-coefficient extension (layer-41 rank-1 residual edit, Qwen2.5-14B-Instruct)

Source runs: `L41_main` (paper, alpha <= 4) and `L41_dose_ext` (alpha in {6, 8, 12, 16}; generic leg over {0.5, ..., 16}). Same held-out prompts, same `--control_seed 99`. Rank-1 direction only; the 8-dim swap operator is off-manifold by alpha = 4 and was not extended.

Metric definitions are in the module docstring of `scripts/circuit/analyze_dose_extension.py`. Applied edit norm is alpha x 53.6; the clean residual norm at this site is about 390 (stage-1 capture, final token).

## Quoted numbers

- True-belief injection, alternative-location rate (on-manifold flip) at alpha 1/2/4/6/8: 9.3% / 16.4% / 31.1% / 10.7% / 0.0%. Peak at alpha = 4.
- True-belief injection, off-manifold rate at alpha 4/6/8: 3.6% / 58.7% / 96.9%.
- Generic prompts (n = 225), target direction inject, KL / top-1 unchanged: alpha 2: 0.015 nats / 99.1%; alpha 4: 0.049 nats / 98.7%; alpha 6: 0.681 nats / 91.6%; alpha 8: 2.199 nats / 82.7%.
- Generic prompts, norm-matched random direction: alpha 2: 0.012 nats / 98.7%; alpha 4: 0.089 nats / 97.3%; alpha 6: 0.400 nats / 93.3%; alpha 8: 0.971 nats / 87.1%.
- Generic clean confidence: median p(top-1) = 1.000, fraction with p(top-1) > 0.5 = 1.000.

## Clean-pass reproduction (L41_dose_ext vs L41_main, same prompts)

| prompt_set | n | mean_signed_dp | mean_abs_dp | max_abs_dp | n_abs_dp_gt_0_1 | top1_agreement |
|---|---|---|---|---|---|---|
| basic_ab | 225 | 0.0022 | 0.0095 | 0.4002 | 3 | 0.9867 |
| code_alias | 150 | 0.0028 | 0.0103 | 0.1078 | 1 | 0.9933 |
| false_belief | 225 | -0.0004 | 0.0041 | 0.1889 | 2 | 0.9867 |
| true_belief | 225 | 0.0004 | 0.0081 | 0.3016 | 6 | 0.9911 |

## True belief, inject

Accuracy (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 92.9 |  |  |  |  |  |  |  |  |
| heads_ctl |  | 94.7 | 96.4 | 97.8 | 97.8 | 93.8 | 85.8 | 56.9 | 26.2 |
| random |  | 92.4 | 92.4 | 91.6 | 85.8 | 83.1 | 72.9 | 48.4 | 28.0 |
| shuffled |  | 92.4 | 92.0 | 88.0 | 78.2 | 63.6 | 46.7 | 12.0 | 7.1 |
| target_rank1 |  | 91.6 | 89.8 | 82.7 | 64.4 | 30.2 | 3.1 | 0.4 | 0.0 |

Alternative-location rate (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 6.2 |  |  |  |  |  |  |  |  |
| heads_ctl |  | 4.4 | 2.2 | 0.4 | 0.0 | 1.3 | 3.1 | 3.1 | 2.2 |
| random |  | 6.7 | 6.7 | 7.6 | 13.3 | 15.1 | 20.9 | 20.0 | 13.8 |
| shuffled |  | 6.7 | 7.1 | 11.6 | 21.3 | 25.8 | 20.0 | 12.4 | 4.9 |
| target_rank1 |  | 7.6 | 9.3 | 16.4 | 31.1 | 10.7 | 0.0 | 0.0 | 0.0 |

Off-manifold rate (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.0 |  |  |  |  |  |  |  |  |
| heads_ctl |  | 0.0 | 0.0 | 0.0 | 0.4 | 3.1 | 9.3 | 38.2 | 69.8 |
| random |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 4.4 | 29.8 | 56.9 |
| shuffled |  | 0.0 | 0.0 | 0.0 | 0.4 | 10.7 | 32.4 | 74.7 | 88.0 |
| target_rank1 |  | 0.0 | 0.0 | 0.0 | 3.6 | 58.7 | 96.9 | 99.6 | 100.0 |

KL(steered || clean), nats

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.000 |  |  |  |  |  |  |  |  |
| heads_ctl |  | 0.020 | 0.073 | 0.169 | 0.296 | 1.117 | 3.103 | 13.043 | 24.587 |
| random |  | 0.002 | 0.006 | 0.027 | 0.146 | 0.487 | 2.056 | 9.783 | 22.544 |
| shuffled |  | 0.008 | 0.041 | 0.227 | 1.002 | 3.599 | 10.788 | 29.141 | 34.007 |
| target_rank1 |  | 0.011 | 0.052 | 0.261 | 2.218 | 16.615 | 29.542 | 31.873 | 32.891 |

## False belief, delete

Accuracy (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 96.9 |  |  |  |  |  |  |  |  |
| heads_ctl |  | 96.9 | 96.4 | 95.6 | 93.3 | 91.1 | 82.7 | 56.0 | 31.6 |
| random |  | 96.4 | 95.6 | 95.6 | 94.7 | 94.2 | 90.7 | 44.0 | 13.3 |
| shuffled |  | 96.4 | 94.7 | 94.2 | 92.0 | 74.2 | 48.4 | 22.2 | 8.0 |
| target_rank1 |  | 97.8 | 97.8 | 97.3 | 94.2 | 77.3 | 40.0 | 11.6 | 6.7 |

Alternative-location rate (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.9 |  |  |  |  |  |  |  |  |
| heads_ctl |  | 0.9 | 0.9 | 1.3 | 0.9 | 0.9 | 0.0 | 0.0 | 0.0 |
| random |  | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 1.3 | 1.3 | 0.4 |
| shuffled |  | 0.9 | 0.9 | 1.3 | 1.8 | 1.8 | 2.2 | 2.2 | 2.2 |
| target_rank1 |  | 1.3 | 1.3 | 2.7 | 5.3 | 19.1 | 27.1 | 3.6 | 0.9 |

Off-manifold rate (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 2.2 |  |  |  |  |  |  |  |  |
| heads_ctl |  | 2.2 | 2.7 | 3.1 | 5.8 | 8.0 | 17.3 | 44.0 | 68.4 |
| random |  | 2.7 | 3.6 | 3.6 | 4.4 | 4.9 | 8.0 | 54.7 | 86.2 |
| shuffled |  | 2.7 | 4.4 | 4.4 | 6.2 | 24.0 | 49.3 | 75.6 | 89.8 |
| target_rank1 |  | 0.9 | 0.9 | 0.0 | 0.4 | 3.6 | 32.4 | 84.9 | 92.4 |

KL(steered || clean), nats

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.000 |  |  |  |  |  |  |  |  |
| heads_ctl |  | 0.003 | 0.010 | 0.032 | 0.157 | 0.822 | 2.708 | 10.657 | 19.074 |
| random |  | 0.001 | 0.002 | 0.013 | 0.118 | 0.437 | 1.632 | 15.847 | 29.600 |
| shuffled |  | 0.006 | 0.022 | 0.061 | 0.461 | 3.802 | 12.152 | 24.827 | 30.900 |
| target_rank1 |  | 0.009 | 0.027 | 0.075 | 0.699 | 4.305 | 14.284 | 30.333 | 33.629 |

## True belief, delete (precondition absent)

Accuracy (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 92.9 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 94.2 | 94.7 | 96.0 | 94.2 | 88.0 | 63.6 | 12.9 | 3.1 |

Alternative-location rate (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 6.2 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 4.9 | 4.4 | 3.6 | 3.6 | 2.7 | 3.6 | 6.2 | 3.6 |

Off-manifold rate (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.0 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 0.0 | 0.0 | 0.0 | 2.2 | 9.3 | 31.6 | 79.6 | 92.0 |

KL(steered || clean), nats

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.000 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 0.005 | 0.015 | 0.050 | 0.355 | 2.135 | 8.587 | 26.096 | 31.759 |

## False belief, inject (precondition absent)

Accuracy (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 96.9 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 95.1 | 94.2 | 92.0 | 83.1 | 47.6 | 5.8 | 0.0 | 0.0 |

Alternative-location rate (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.9 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 0.9 | 0.9 | 0.9 | 0.9 | 0.4 | 0.4 | 0.0 | 0.0 |

Off-manifold rate (%)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 2.2 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 4.0 | 4.9 | 7.1 | 16.0 | 52.0 | 93.8 | 100.0 | 100.0 |

KL(steered || clean), nats

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.000 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 0.007 | 0.025 | 0.111 | 1.190 | 10.056 | 23.790 | 27.795 | 28.838 |

## basic_ab (belief-free, target inject)

p(correct)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.900 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 0.887 | 0.859 | 0.786 | 0.634 | 0.370 | 0.037 | 0.000 | 0.000 |

KL

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.000 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 0.020 | 0.128 | 0.705 | 3.318 | 12.599 | 28.360 | 33.082 | 33.597 |

top-1 unchanged (extension doses only)

| direction | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|
| target_rank1 | 0.413 | 0.031 | 0.000 | 0.000 |

## code_alias (belief-free, target inject)

p(correct)

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.766 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 0.756 | 0.745 | 0.722 | 0.695 | 0.677 | 0.613 | 0.107 | 0.005 |

KL

| direction | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.000 |  |  |  |  |  |  |  |  |
| target_rank1 |  | 0.003 | 0.010 | 0.041 | 0.155 | 0.386 | 2.439 | 26.140 | 35.227 |

top-1 unchanged (extension doses only)

| direction | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|
| target_rank1 | 0.827 | 0.753 | 0.220 | 0.053 |

## Generic belief-free prompts

KL(steered || clean), nats

| mode_dir | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean/clean | 0.000 |  |  |  |  |  |  |  |  |
| delete/target_rank1 |  | 0.005 | 0.022 | 0.071 | 0.313 | 0.656 | 1.856 | 12.082 | 33.139 |
| inject/heads_ctl |  | 0.001 | 0.002 | 0.011 | 0.113 | 0.364 | 1.643 | 10.903 | 24.183 |
| inject/random |  | 0.001 | 0.003 | 0.012 | 0.089 | 0.400 | 0.971 | 3.318 | 21.956 |
| inject/shuffled |  | 0.002 | 0.005 | 0.025 | 0.185 | 1.580 | 6.442 | 28.961 | 46.419 |
| inject/target_rank1 |  | 0.002 | 0.007 | 0.015 | 0.049 | 0.681 | 2.199 | 21.633 | 42.027 |

Top-1 unchanged (fraction)

| mode_dir | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean/clean | 1.000 |  |  |  |  |  |  |  |  |
| delete/target_rank1 |  | 0.996 | 0.987 | 0.973 | 0.911 | 0.880 | 0.867 | 0.720 | 0.422 |
| inject/heads_ctl |  | 1.000 | 1.000 | 0.982 | 0.964 | 0.951 | 0.893 | 0.578 | 0.244 |
| inject/random |  | 0.996 | 0.996 | 0.987 | 0.973 | 0.933 | 0.871 | 0.773 | 0.516 |
| inject/shuffled |  | 0.996 | 0.996 | 0.991 | 0.969 | 0.867 | 0.689 | 0.116 | 0.000 |
| inject/target_rank1 |  | 1.000 | 0.996 | 0.991 | 0.987 | 0.916 | 0.827 | 0.391 | 0.178 |

Mean p(top-1)

| mode_dir | a=0 | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|---|
| clean/clean | 0.990 |  |  |  |  |  |  |  |  |
| delete/target_rank1 |  | 0.987 | 0.985 | 0.978 | 0.977 | 0.972 | 0.937 | 0.680 | 0.248 |
| inject/heads_ctl |  | 0.988 | 0.986 | 0.983 | 0.980 | 0.970 | 0.933 | 0.729 | 0.526 |
| inject/random |  | 0.990 | 0.990 | 0.987 | 0.986 | 0.965 | 0.958 | 0.880 | 0.479 |
| inject/shuffled |  | 0.991 | 0.991 | 0.987 | 0.974 | 0.937 | 0.845 | 0.641 | 0.585 |
| inject/target_rank1 |  | 0.991 | 0.992 | 0.993 | 0.987 | 0.957 | 0.929 | 0.743 | 0.698 |

Top-1 unchanged by family, target inject

| family | a=0.5 | a=1 | a=2 | a=4 | a=6 | a=8 | a=12 | a=16 |
|---|---|---|---|---|---|---|---|---|
| antonym | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.79 | 0.05 | 0.00 |
| arithmetic | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.72 |
| calendar | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.71 | 0.00 |
| capital | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 | 0.38 | 0.10 | 0.00 |
| count_letters | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.77 |
| first_letter | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.28 | 0.00 |
| multiple_choice | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.05 | 0.00 |
| open | 1.00 | 0.93 | 0.93 | 0.86 | 0.86 | 0.71 | 0.21 | 0.00 |
| python | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.90 | 0.75 | 0.35 |
| translate | 1.00 | 1.00 | 0.95 | 0.95 | 0.84 | 0.53 | 0.11 | 0.00 |
| yes_no | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.96 | 0.04 | 0.00 |
