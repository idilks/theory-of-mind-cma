"""
Prompt generation modules for CMA experiments.
"""

import logging

from .base import PromptGenerator
from .identity_rules import IdentityRulesGenerator
from .unified_tom import UnifiedTomGenerator
from .code_generator import CodeGenerator

# Own module logger rather than borrowing cma's: cma imports this package, so
# `from cma import logger` had to be deferred into the function body and made
# the package unimportable outside a cma run. Handlers/level still come from
# cma's logging.basicConfig(), so emitted output is unchanged apart from the
# logger name.
logger = logging.getLogger(__name__)


def get_prompt_generator(config, tokenizer) -> PromptGenerator:
    """Factory function to create appropriate prompt generator."""
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