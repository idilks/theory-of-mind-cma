# Transformer Causal Analysis

causal mediation analysis for transformer attention mechanisms in in-context learning and theory of mind reasoning.

## quick start

```bash
# basic causal analysis
python causal_analysis.py --model_type "8B" --prompt_num 100 --base_rule "ABA"

# theory of mind analysis  
python codebase/tasks/identity_rules/cma.py \
  --use_tom_prompts \
  --context_type "abstract" \
  --base_rule "ABA" \
  --prompt_num 50 \
  --model_type "Llama-3.2-1B"

# behavioral evaluation (no mechanistic analysis)
python behavioral/behavioral_eval.py --config_type full_comparison
```

## architecture

- **causal_analysis.py**: main activation patching for rule learning (ABA vs ABB patterns)
- **codebase/tasks/identity_rules/cma.py**: refactored modular cma pipeline
- **behavioral/**: pure behavioral evaluation with wandb tracking
- **codebase/**: integrated LLMSymbMech framework for symbolic processing analysis

## key experiments

**activation patching**: systematically patch activations between prompt conditions to identify causal components for rule following

**theory of mind**: adapted cma framework from token patterns to false belief reasoning - identifies belief tracking vs location retrieval heads

**behavioral evaluation**: temperature sweeps across models for pure tom performance without mechanistic overhead

## models supported

llama-3.1 (8B, 70B, instruct variants), qwen2.5 (7B-72B), gpt-2, gemma-2. automatic embedding resize handles tokenizer mismatches.

## setup

1. get hf token: https://huggingface.co/settings/tokens
2. `export HF_TOKEN="your_token"`
3. `pip install transformers transformer_lens torch wandb`

## output

results saved as heatmaps showing causal importance by layer×head. raw tensors and prompt files included for replication.