# Provenance of the head-level geometry / steering / cross-patch results

Copied in August 2026 from the PI's handoff bundle, which carried derived CSVs
only (no activations, no scripts).

## Renames applied on copy-in, and why

The writers (`state_geometry_alignment.py`, `state_geometry_threeway.py`,
`steer_state_axes.py`) used model-neutral output dirs, so the Gemma runs
overwrote the Qwen runs in place. Model identity was recovered from head-set
sizes and re-derived locally with `head_sets_source.build()`:

    Qwen2.5-14B-Instruct   div_unique 22, div_shared 10, dual_sig 10,
                           photo_C2 21, address 32
    gemma-2-27b-it         div_unique 14, div_shared 7, photo_C2 25, no address

| bundle path | copied to | model | evidence |
|---|---|---|---|
| `state_geometry_pre_dual/` | `state_geometry_qwen/` | Qwen | n = 22/10, `address` present, `auc` 0.8359 = the §G anchor 0.836 |
| `state_geometry/` | `state_geometry_gemma/` | Gemma | n = 14/7/25, no `address` row |
| `state_geometry_threeway/` | `state_geometry_threeway_gemma/` | Gemma | n = 14/7/25; `belief_code_corr` 0.4373 and locality 0.4989 = §GM's +0.437 / 0.499. The Qwen §SG4 threeway (bench 0.788, belief↔photo 0.528, belief↔code 0.271) is **not in the bundle** — docs only |
| `steering/` | `steering_qwen/` | Qwen | §S magnitudes (u_cm → code +42.9, u_tom → tom +20.9); §GM7 reports ~7× lower gain for Gemma |
| `steering_photo_qwen/` | `steering_photo_qwen/` | Qwen | belief rows identical to `steering_qwen/`'s tom rows |
| `steering_photo/` | **dropped** | Qwen | byte-identical (md5) to `steering_photo_qwen/` on all three files |
| `cross_patch/` | `cross_patch/` | Qwen | `head_sets.json` = 22/10/32 |

`state_geometry_alignment.py`, `state_geometry_threeway.py` and
`steer_state_axes.py` now suffix their output dirs and figure names with
`dims.tag(model_type)`, so a rerun writes `state_geometry_qwen/` /
`state_geometry_gemma/` rather than clobbering. `cross_patch_experiment.py`
still writes a model-neutral `cross_patch/` — no Gemma run exists yet, but the
same fix is needed before one does.
