"""
Base classes and utilities for prompt generation.
Simplified parsing focused on <loc> tag extraction.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Tuple

def extract_location_from_response(response: str, loc1: str = None, loc2: str = None) -> str:
    """
    Extract location from model response.
    Priority: <loc>X</loc> tags > known locations > first capitalized phrase
    this only works in the single-answer setting currently.
    """
    # 1. Try to extract from <loc> tags (handles both <loc>X</loc> and partial <loc>X)
    # For completion format: response starts with location and ends with </loc>
    # For instruction format: response has <loc>location</loc> anywhere
    
    # First try: completion format (starts with location, ends with </loc>)
    completion_match = re.search(r"^\s*([^<]+?)\s*</loc>", response, re.IGNORECASE)
    if completion_match:
        extracted = completion_match.group(1).strip()
        if extracted:
            return extracted
    
    # Second try: standard format <loc>content</loc>
    tag_match = re.search(r"<loc>(.*?)(?:</loc>|$)", response, re.IGNORECASE | re.DOTALL)
    if tag_match:
        extracted = tag_match.group(1).strip()
        # Clean up common issues
        extracted = re.sub(r'\s+', ' ', extracted)  # normalize whitespace
        extracted = extracted.split('.')[0].strip()  # stop at first period
        extracted = extracted.split(',')[0].strip()  # stop at first comma
        # Remove incomplete closing tags like </
        extracted = re.sub(r'</?$', '', extracted).strip()
        # Remove common trailing words
        for word in ['since', 'because', 'as', 'where', 'that']:
            if extracted.lower().endswith(' ' + word):
                extracted = extracted[:-(len(word)+1)].strip()
                print(f"Trimmed trailing word '{word}' from extracted location. it is now '{extracted}'")
        if extracted != "":
            return extracted
    
    # 3. Fallback: extract first location-like phrase (capitalized or after preposition)
    # Look for: "on X", "at X", "in X", "near X", "by X"
    prep_match = re.search(r'\b(?:on|at|in|near|by|to)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?)(?:\.|,|$)', response)
    if prep_match:
        return prep_match.group(1).strip()
    
    # Look for capitalized phrases (2-3 words max)
    cap_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', response)
    if cap_match:
        return cap_match.group(1).strip()
    
    return ""


def normalize_loc(s: str) -> str:
    """Normalize location string for comparison."""
    if not s:
        return ""
    
    normalized = s.strip().lower()
    
    # Remove articles from beginning
    for article in ["the ", "a ", "an "]:
        if normalized.startswith(article):
            normalized = normalized[len(article):]
            break
    
    return normalized


def locations_match(pred: str, expected: str) -> bool:
    """Check if predicted location matches expected location."""
    if not pred or not expected:
        return False
    
    pred_norm = normalize_loc(pred)
    expected_norm = normalize_loc(expected)
    
    # Exact match after normalization
    if pred_norm == expected_norm:
        return True
    
    # Contains match (handles "Main Street" in "on Main Street")
    if expected_norm in pred_norm or pred_norm in expected_norm:
        return True
    
    return False


# Dual-answer support (for future --ask_world mode)
def extract_dual_answers(text: str) -> Tuple[str, str, bool]:
    """
    Extract belief and world answers from dual-answer format.
    Returns (belief, world, malformed)
    """
    belief_pattern = re.compile(r"(?:agent thinks|where.*thinks).*?<loc>(.*?)</loc>", re.I | re.DOTALL)
    world_pattern = re.compile(r"(?:actual|real).*?<loc>(.*?)</loc>", re.I | re.DOTALL)
    
    belief_match = belief_pattern.search(text)
    world_match = world_pattern.search(text)
    
    belief = belief_match.group(1).strip() if belief_match else ""
    world = world_match.group(1).strip() if world_match else ""
    
    malformed = (belief == "" or world == "")
    return belief, world, malformed


def wrap_dual_schema(scenario: str) -> str:
    """Wrap scenario in dual-answer schema for --ask_world mode."""
    return (
        f"{scenario}\n"
        "answer in this exact schema:\n"
        "where the agent thinks: <loc></loc>\n"
        "actual location: <loc></loc>\n"
    )


class PromptGenerator(ABC):
    """Abstract base class for prompt generators."""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
    @abstractmethod
    def generate_prompts(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str]]:
        """
        Generate prompt pairs for CMA analysis.
        
        Returns:
            prompt_pairs: List of (base_prompt, exp_prompt) tuples
            correct_ans_pairs: List of (base_ans, exp_ans) tuples
            causal_ans_list: List of expected answers after patching
        """
        pass


def classify_belief_prediction(predicted: str, expected_belief: str, alternative_belief: str) -> str:
    """Classify belief prediction as correct, opposite, or third."""
    predicted = normalize_loc(predicted)
    expected = normalize_loc(expected_belief)
    alternative = normalize_loc(alternative_belief)
    
    if locations_match(predicted, expected):
        return "correct"
    elif locations_match(predicted, alternative):
        return "opposite"
    else:
        return "third"


def classify_world_prediction(predicted: str, expected_world: str, expected_belief: str) -> str:
    """Classify world prediction as correct, belief_confusion, or third."""
    predicted = normalize_loc(predicted)
    world_state = normalize_loc(expected_world)
    belief_state = normalize_loc(expected_belief)
    
    if locations_match(predicted, world_state):
        return "correct"
    elif locations_match(predicted, belief_state):
        return "belief_confusion"
    else:
        return "third"