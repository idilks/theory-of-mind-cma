## Project Overview

This is a research project focused on causal analysis of transformer language models, specifically analyzing how different attention mechanisms contribute to in-context learning patterns. The codebase implements activation patching techniques to understand causal relationships in model behavior.

## Core Architecture

### Main Components

- **cma.py**: Main experimental script containing the core causal mediation analysis implementation
- **utils.py**: Utility functions for plotting, data manipulation, and model configuration

### Key Classes and Functions

- `CustomHookTransformer`: Extended transformer_lens.HookedTransformer with custom generation and caching capabilities
- `activation_patching()`: Core function that performs causal mediation by patching activations between different prompt conditions
- `ablate_head()` and `ablate_layer()`: Functions for systematically ablating different components (attention heads or residual stream layers)
- `generate_prompts()`: Creates structured prompt pairs for testing different rule patterns (ABA vs ABB)

## Running the Code

### Basic Execution
```bash
python cma.py --model_type "8B" --prompt_num 100 --base_rule "ABA"
```

### Key Command Line Arguments

- `--model_type`: Model variant (8B, 70B, 8B-Instruct, etc.)
- `--activation_name`: Which activation to patch (z, q, k, v, resid_pre, etc.)
- `--base_rule`: Base rule pattern (ABA or ABB)
- `--prompt_num`: Number of prompts to generate
- `--sample_num`: Number of samples for causal mediation analysis
- `--token_pos_list`: Token positions for patching (default: [-1])
- `--generate`: Use generation-based evaluation instead of logit differences
- `--eval_generation`: Filter prompts by generation accuracy

### Configuration Requirements

- Set HF_TOKEN in utils.py for Hugging Face model access (both main utils.py and codebase/utils.py)
- Configure XDG_CACHE_HOME for model caching
- Ensure adequate GPU memory for transformer models

## Research Context

### Experimental Design
The code implements causal mediation analysis to understand how transformer attention mechanisms process in-context learning rules. It compares model behavior on two rule types:
- ABA pattern: First and third tokens match
- ABB pattern: Second and third tokens match

## LLMSymbMech Integration (Now in codebase/)

The `codebase/` folder builds upon the ICML 2025 "Emergent Symbolic Mechanisms" codebase (copied from LLMSymbMech), which implements a three-stage symbolic processing framework. This extends our causal analysis approach with targeted mechanism identification.

### Three-Stage Symbolic Processing

The LLMSymbMech approach identifies three distinct attention head types:

1. **Symbol abstraction heads**: Identify relations between input tokens and represent them using abstract variables
2. **Symbolic induction heads**: Perform sequence induction over abstract variables
3. **Retrieval heads**: Predict tokens by retrieving values associated with predicted abstract variables

### Dataset Integration

**LLMSymbMech Dataset Structure**:
- `datasets/vocab/`: Curated english-only tokens per model family
- `datasets/cma_scores/`: Pre-computed causal scores and significant heads
- `datasets/*_correct_common_tokens_*.txt`: High-accuracy token subsets (e.g., 1378 tokens for Llama-3.1-70B)
- Pre-generated prompt files for exact replication

### Activation Patching Process
1. Generate prompt pairs that differ in rule structure or token order
2. Run forward passes to collect activations
3. Systematically patch activations from one condition to another
4. Measure changes in model predictions to identify causal components

### Output Analysis
Results are saved as heatmaps showing the causal importance of different model components (layers × heads or layers × positions) for rule following behavior.

## Dependencies

The project requires:
- transformers (Hugging Face)
- transformer_lens
- torch
- numpy
- matplotlib/seaborn for visualization
- scipy for statistical analysis

## File Organization

Results are automatically organized in hierarchical folders:
- `causal_mediation_results_full_more_models/[model_name]/[experiment_type]/[rule_configuration]/`
- Includes input prompts, activation caches, and analysis plots

## Setup Instructions

### Prerequisites
1. **Python Environment**: Python 3.8+ with pip
2. **GPU Access**: CUDA-capable GPU recommended for transformer models
3. **HuggingFace Account**: Required for model access and API

### Installation Steps

1. **Environment Activation** (Current Setup):
   ```bash
   # RECOMMENDED: Use wrapper script (always works)
   ./run_with_env.sh python script_name.py
   
   # INTERACTIVE: Use activation script for terminal sessions
   source activate_tom_env.sh        # Linux/WSL/Mac
   activate_tom_env.bat              # Windows Command Prompt
   
   # DIRECT: Run Python directly 
   .conda/python.exe script_name.py
   

   **Note**: The current `.conda` directory contains a full conda distribution rather than a proper environment. This works but isn't standard practice.

2. **Alternative: Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Authentication**:
   - Get HuggingFace token: https://huggingface.co/settings/tokens
   - Set environment variable: `export HF_TOKEN="your_token_here"`
   - Or edit `utils.py` line 10 to replace the placeholder

4. **HPC Configuration** (Dartmouth users):
   - Uncomment and modify cache path in `utils.py`:
     ```python
     XDG_CACHE_HOME = "/scratch/gpfs/yourusername/.cache"
     ```


```bash
# 1. Backup current working setup
mv .conda .conda_backup

# 2. Create proper conda environment (if conda available)
conda create -n tom_analysis python=3.12
conda activate tom_analysis
pip install -r requirements.txt

# 3. OR create virtual environment
python -m venv tom_env
source tom_env/bin/activate  # Linux/Mac
# OR tom_env\Scripts\activate  # Windows
pip install -r requirements.txt

# 4. Update scripts to use proper environment name
# conda activate tom_analysis
```


## HPC Troubleshooting


**Quick Diagnosis**:
```bash
# On HPC, run the diagnostic script
python diagnose_hpc_env.py
```


**Alternative Models** (if Llama fails):
```bash
# Use GPT-2 for testing
python causal_analysis.py --model_type "gpt2"

# Or other supported models
python bigtom_api_test.py --model_name "microsoft/DialoGPT-medium"
```

### Running the Causal Analysis

**Basic Examples** (Working Commands):

```bash
# Local testing with small model
./run_with_env.sh python causal_analysis.py \
  --model_type "Llama-3.2-1B" \
  --prompt_num 10 \
  --base_rule "ABA" \
  --activation_name "z" \
  --exp_swap_1_2_question \
  --sample_num 5

# HPC with larger model  
python causal_analysis.py \
  --model_type "8B-Instruct" \
  --prompt_num 100 \
  --base_rule "ABA" \
  --activation_name "z" \
  --exp_swap_1_2_question \
  --sample_num 50 \
  --generate

# Quick test run
python causal_analysis.py \
  --model_type "Llama-3.2-1B" \
  --prompt_num 5 \
  --base_rule "ABA" \
  --activation_name "resid_pre" \
  --exp_swap_1_2_question \
  --sample_num 3
```

**Key Parameters**:
- `--model_type`: "8B", "8B-Instruct", "Llama-3.2-1B", "Qwen2.5-7B"
- `--activation_name`: "z", "q", "k", "v", "resid_pre", "resid_post"  
- `--base_rule`: "ABA" or "ABB"
- `--exp_swap_1_2_question`: Required flag for prompt generation
- `--generate`: Use generation accuracy instead of logit differences
- `--sample_num`: Number of samples to patch (use small numbers for testing)

**Testing Dependencies**:
```bash
# Verify all dependencies work before running experiments
python test_causal_analysis_deps.py
```

### Running LLMSymbMech Examples

**Test Symbol Abstraction Head Detection**:
```bash
cd codebase/tasks/identity_rules

# Quick test with small model
python cma.py \
  --model_type Llama-3.2-1B \
  --activation_name z \
  --context_type abstract \
  --token_pos_list 4 10 \
  --base_rule ABA \
  --prompt_num 50 \
  --min_valid_sample_num 10 \
  --in_context_example_num 2 \
  --seed 0 \
  --low_prob_threshold 0.9 \
  --do_shuffle

# Full analysis (requires 70B model)
python cma.py \
  --model_type Llama-3.1-70B \
  --activation_name z \
  --context_type abstract \
  --token_pos_list 4 10 \
  --base_rule ABA \
  --prompt_num 1000 \
  --min_valid_sample_num 20 \
  --in_context_example_num 10 \
  --seed 0 \
  --low_prob_threshold 0.9 \
  --do_shuffle \
  --ungroup_grouped_query_attention \
  --eval_metric gen_acc \
  --load_generation_config \
  --sample_size 4 \
  --n_devices 2
```

**Compare with Pre-computed Results**:
```bash
# Load their pre-identified significant heads
python -c "
import torch
from utils import get_head_list, llama31_70B_significant_head_dict
heads, scores = get_head_list('symbol_abstraction_head')
print(f'Found {len(heads)} significant symbol abstraction heads')
print(f'Top 5 heads: {heads[:5]}')
"
```

**Output Files**:
- **Heatmaps**: Visual results showing causal importance by layer/head
- **Data files**: Raw pytorch tensors with numerical results  
- **Prompts**: Generated input prompts used for experiments
- **Answers**: Expected vs actual answers for evaluation

## Theory of Mind Causal Mediation Analysis


### Theoretical Framework

**Abstract Context (Belief Tracking Heads)**:
- Base: False belief scenario (agent misses object movement)
- Exp: True belief scenario (agent witnesses movement)  
- Hypothesis: Patching "belief tracking heads" should convert true belief reasoning to false belief

**Token Context (Location Retrieval Heads)**:
- Base & Exp: Same false belief scenario with different phrasing
- Hypothesis: Patching "location retrieval heads" should preserve literal location answers

### Running Theory of Mind CMA

**Basic Example**:
```bash
# Test belief tracking heads (abstract context)
python codebase/tasks/identity_rules/cma.py \
  --use_tom_prompts \
  --context_type "abstract" \
  --base_rule "ABA" \
  --prompt_num 50 \
  --model_type "Llama-3.2-1B" \
  --activation_name "z" \
  --token_pos_list -1 \
  --sample_size 4 \
  --min_valid_sample_num 10

# Test location retrieval heads (token context)  
python codebase/tasks/identity_rules/cma.py \
  --use_tom_prompts \
  --context_type "token" \
  --base_rule "ABA" \
  --prompt_num 50 \
  --model_type "Llama-3.2-1B" \
  --activation_name "z"
```

**Key Parameters**:
- `--use_tom_prompts`: Switch from identity rules to theory of mind prompts
- `--context_type`: "abstract" (belief tracking) or "token" (location retrieval)
- `--base_rule`: "ABA" (false belief) or "ABB" (true belief) base scenarios
- `--tom_locations_file`: Location phrases file (default: tom_datasets/locations.txt)

**Testing Prompt Generation**:
```bash
# Verify tom prompt logic before running full experiments
python test_tom_cma.py
```

### Theory of Mind Dataset Structure

**Location Phrases** (`tom_datasets/locations.txt`):
- 150+ spatial location phrases like "under the table", "next to the shelf"
- Used to generate diverse false belief scenarios
- Each scenario uses 2 locations (original + moved location)

**Example Prompt Pairs**:

*Abstract Context (ABA/False Belief):*
- Base: "object is <loc>kitchen</loc>. agent leaves room. object moves to <loc>garden</loc>. agent returns and looks where?" → kitchen
- Exp: "object is <loc>kitchen</loc>. object moves to <loc>garden</loc>. agent leaves room. agent returns and looks where?" → garden  
- Causal: After patching belief heads, exp should answer kitchen (false belief)

*Token Context (Location Retrieval):*
- Base: "object is <loc>kitchen</loc>. agent leaves room. object moves to <loc>garden</loc>. agent returns and looks where?" → kitchen
- Exp: "object is <loc>kitchen</loc>. agent leaves room. object moves to <loc>garden</loc>. where does agent look?" → kitchen
- Causal: Should stay kitchen (literal retrieval unchanged)

### Expected Results Structure

```
results/identity_rules/cma/
└── [model_name]/
    └── abstract_context/  # or token_context
        └── base_rule_ABA_exp_rule_ABB/  # or ABB_exp_rule_ABA
            └── z_seed_[N]_shuffle_[bool]/
                ├── logit/  # or generate/
                │   └── sample_num_[N]_gen_acc_0.9/
                │       ├── group_heads_False/
                │       │   └── token_pos_[-1]/
                │       │       ├── heatmap.png
                │       │       └── causal_scores.pt
                │       ├── base_prompt_[N].txt
                │       └── exp_prompt_[N].txt
                └── base_input_prompts_[N].txt
```

### Integration with Existing Framework

The theory of mind implementation reuses the entire CMA pipeline:
- **Same activation patching logic**: `ablate_head()` and `ablate_layer()`
- **Same evaluation metrics**: Generation accuracy or logit differences
- **Same filtering process**: Only analyzes prompts where model gets both answers correct
- **Same output format**: Heatmaps showing causal importance by layer/head

This enables direct comparison between:
1. **Identity rule mechanisms** (ABA/ABB token patterns)  
2. **Theory of mind mechanisms** (false/true belief reasoning)

Both using identical causal mediation methodology.

## BIGToM Theory of Mind Evaluation

### New Addition: BIGToM API Testing Script

The `bigtom_api_test.py` script evaluates Theory of Mind reasoning using the BIGToM dataset via Hugging Face Inference API.

### Running BIGToM Evaluation

```bash
# Basic evaluation
python bigtom_api_test.py \
  --model_name "meta-llama/Llama-2-7b-chat-hf" \
  --sample_size 100

# Advanced evaluation with chain-of-thought
python bigtom_api_test.py \
  --model_name "meta-llama/Llama-2-7b-chat-hf" \
  --method "chain_of_thought" \
  --sample_size 200
```

### BIGToM Features
- **API Integration**: Works with Hugging Face Inference API
- **Multiple Prompting Methods**: Zero-shot, chain-of-thought, multiple-choice
- **Automatic Data Generation**: Creates sample ToM scenarios if dataset unavailable
- **Comprehensive Analysis**: Accuracy metrics, confidence scoring, visualizations
- **Results Export**: JSON and CSV formats with timestamp

### BIGToM Configuration
- Set `HF_TOKEN` environment variable for API access
- Results saved to `results/` directory
- Automatic retry logic for API failures
- Cost-aware with configurable sample sizes

### File Structure
```
tom_dev/
├── causal_analysis_paper_simple_v2.py  # Main causal analysis
├── bigtom_api_test.py                  # BIGToM evaluation  
├── utils.py                           # Shared utilities
├── requirements.txt                   # Dependencies
├── llama31_english_vocab.txt         # Token vocabulary
├── simple_copy_to_hpc.bat           # File transfer script
├── data/                            # Generated datasets
└── results/                         # Evaluation outputs
```



