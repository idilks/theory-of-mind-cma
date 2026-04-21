"""
Prompt generation modules for CMA experiments.
"""

from .base import PromptGenerator
from .identity_rules import IdentityRulesGenerator
from .unified_tom import UnifiedTomGenerator
from .code_generator import CodeGenerator

def get_prompt_generator(config, tokenizer) -> PromptGenerator:
    """Factory function to create appropriate prompt generator."""
    from cma import logger
    if config.prompts.context_type == "code":
        logger.info("DEBUG: Selected CodeGenerator")
        return CodeGenerator(config, tokenizer)
    logger.info(f"DEBUG: Selecting prompt generator - use_template_system={getattr(config.prompts, 'use_template_system', 'MISSING')}")
    if hasattr(config.prompts, 'use_template_system') and config.prompts.use_template_system:
        logger.info("DEBUG: Selected UnifiedTomGenerator")
        return UnifiedTomGenerator(config, tokenizer)
    else:
        logger.info("DEBUG: Selected IdentityRulesGenerator")
        return IdentityRulesGenerator(config, tokenizer)