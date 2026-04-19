import torch, os, re
BASE = os.path.join('codebase', 'results', 'causal_analysis', 'cma', 'Qwen', 'Qwen2.5-14B-Instruct')
scores = {}
for root, dirs, files in os.walk(BASE):
    if 'causal_scores.pt' in files and 'patch_before_movement' in root:
        rel = os.path.relpath(root, BASE).replace(os.sep, '/')
        parts = rel.split('/')
        condition = parts[0]
        template = parts[1].replace('template_', '')
        m = re.match(r'base_rule_(\w+)_exp_rule_(\w+)', parts[2])
        if not m:
            continue
        key = (condition, template)
        if key not in scores:
            scores[key] = []
        scores[key].append(1)
for k in sorted(scores.keys()):
    print(k)
