"""
Model loading and setup utilities for CMA experiments.
"""

import logging
import os
import sys
import torch
from transformers import AutoTokenizer, GenerationConfig
import transformer_lens

# Add codebase root to path for utils import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import get_model_id_family, vocab_dict


def _get_hf_token():
    """Return Hugging Face token from either supported environment variable."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")

def setup_hf_authentication():
    """Setup HuggingFace authentication if token is available."""
    hf_token = _get_hf_token()
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("Successfully authenticated with HuggingFace")
        except ImportError:
            logger.warning("huggingface_hub not available for login, using token in requests")
        except Exception as e:
            logger.warning(f"HuggingFace login failed: {e}, using token in requests")

logger = logging.getLogger(__name__)

class CustomHookedTransformer(transformer_lens.HookedTransformer):
    def resize_token_embeddings(self, new_num_tokens: int, id_map=None):
        curr = self.W_E.shape[0]
        if new_num_tokens == curr:
            logger.info(f"no change {curr} = {new_num_tokens}")
            return
        elif new_num_tokens < curr and id_map is None:
            logger.info(f'refusing to shrink token embeddings without id_map')
            return

        logger.info(f'resizing embeddings from {curr} to {new_num_tokens}')
        old_E = self.W_E.data
        d = old_E.shape[1]
        new_E = torch.zeros(new_num_tokens, d, dtype=old_E.dtype, device=old_E.device)
        
        if id_map is None:
            n = min(curr, new_num_tokens)
            new_E[:n] = old_E[:n]
        else:
            for new_i, old_i in id_map.items():
                if 0<= new_i < new_num_tokens and 0<= old_i < curr:
                    new_E[new_i] = old_E[old_i]

        self.W_E.data = new_E
        
        # W_U
        if hasattr(self, 'W_U'):
            old_U = self.W_U.data
            new_U = torch.zeros(d, new_num_tokens, dtype=old_U.dtype, device=old_U.device)
            if id_map is None:
                n = min(curr, new_num_tokens)
                new_U[:n] = old_U[:n]
            else:
                for new_i, old_i in id_map.items():
                    if 0<= new_i < new_num_tokens and 0<= old_i < curr:
                        new_U[:, new_i] = old_U[:, old_i]
            self.W_U.data = new_U
            
        if hasattr(self, 'b_U'):
            old_b = self.b_U.data
            new_b = torch.zeros(new_num_tokens, dtype=old_b.dtype, device=old_b.device)
            if id_map is None:
                n = min(curr, new_num_tokens)
                new_b[:n] = old_b[:n]
            else:
                for new_i, old_i in id_map.items():
                    if 0<= new_i < new_num_tokens and 0<= old_i < curr:
                        new_b[new_i] = old_b[old_i]
            self.b_U.data = new_b
            
            if hasattr(self, 'cfg'):
                self.cfg.d_vocab = new_num_tokens

    def generate_with_hooks(
        self, 
        *model_args,
        fwd_hooks = [],
        bwd_hooks = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        **model_kwargs
    ):
        """
        Support implementing forward (e.g. patching activations) and backward hooks during generation. 
        Adapted from generate() and run_with_hooks() in transformer_lens.
        """
        if len(bwd_hooks) > 0 and reset_hooks_end:
            logging.warning(
                "WARNING: Hooks will be reset at the end of generate_with_hooks. This removes the backward hooks before a backward pass can occur."
            )
  
        with self.hooks(fwd_hooks, bwd_hooks, reset_hooks_end, clear_contexts) as hooked_model:
            return hooked_model.generate(*model_args, **model_kwargs)

class ModelLoader:
    """Handles model loading and setup."""
    
    def __init__(self, model_config, generation_config, prompt_config):
        self.model_config = model_config
        self.generation_config = generation_config
        self.prompt_config = prompt_config
        
    def load_model_and_tokenizer(self):
        """Load model, tokenizer and setup generation config."""
        # Setup HuggingFace authentication first
        setup_hf_authentication()
        
        model_id, model_family = get_model_id_family(self.model_config.model_type)
        logger.info(f"model type: {self.model_config.model_type}, model id: {model_id}")
        
        # Load tokenizer with authentication
        hf_token = _get_hf_token()
        tokenizer_kwargs = {"trust_remote_code": True, "use_fast": True}
        if hf_token:
            tokenizer_kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        torch.set_grad_enabled(False)
        
        # Setup generation kwargs
        generation_kwargs = self._setup_generation_config()
        
        # Load model
        model = self._load_model(model_id, model_family, tokenizer)
        
        # Setup tokenizer embeddings and special tokens
        self._setup_model_embeddings(model, tokenizer)
        eos_token_ids, A_tok_id, B_tok_id = self._setup_special_tokens(model, tokenizer)
        
        return model, tokenizer, generation_kwargs, eos_token_ids, A_tok_id, B_tok_id, model_id
    
    def _setup_generation_config(self):
        """Setup generation configuration."""
        if self.generation_config.load_generation_config:
            config_name = self.generation_config.generation_config_name or f"{self.model_config.model_type}.json"
            gen_config = GenerationConfig.from_pretrained("./generation_config/", config_file_name=config_name)
            logger.info(f"loading generation config from ./generation_config/{config_name}")
            
            return {
                "top_k": gen_config.top_k,
                "top_p": gen_config.top_p,
                "do_sample": gen_config.do_sample,
                "temperature": gen_config.temperature,
                "freq_penalty": gen_config.repetition_penalty
            }
        else:
            logger.warning("Using default multinomial sampling generation config [Not recommended]")
            return {
                "top_k": self.generation_config.top_k,
                "top_p": self.generation_config.top_p,
                "do_sample": self.generation_config.do_sample,
                "temperature": self.generation_config.temperature,
            }
    
    def _load_model(self, model_id, model_family, tokenizer):
        """Load the transformer model."""
        try:
            if model_family == "gpt-oss":
                logger.info(f"Attempting to load {model_id} directly (experimental support)")
                try:
                    from transformers import AutoModelForCausalLM
                    load_kwargs_hf = {
                        "torch_dtype": torch.bfloat16,
                        "device_map": self.model_config.device_map,
                        "trust_remote_code": True,
                    }
                    
                    # Add HF token if available
                    hf_token = _get_hf_token()
                    if hf_token:
                        load_kwargs_hf["token"] = hf_token
                    
                    # For single device, use device instead of device_map to avoid fragmentation
                    if self.model_config.device_map == "cuda" and self.model_config.n_devices == 1:
                        del load_kwargs_hf["device_map"]  
                        load_kwargs_hf["device"] = "cuda"
                    
                    # Add quantization if specified
                    if self.model_config.load_in_8bit:
                        load_kwargs_hf["load_in_8bit"] = True
                        logger.info("Loading model with 8-bit quantization")
                    elif self.model_config.load_in_4bit:
                        load_kwargs_hf["load_in_4bit"] = True
                        logger.info("Loading model with 4-bit quantization")
                    
                    hf_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs_hf)
                    # Use device_map OR device, not both (avoids tensor placement conflicts)
                    load_kwargs = {
                        "tokenizer": tokenizer,
                        "n_devices": self.model_config.n_devices,
                        "torch_dtype": torch.bfloat16,
                        "fold_ln": self.model_config.fold_ln,
                        "center_writing_weights": False,
                        "center_unembed": False,
                    }
                    
                    # Don't pass device_map when using pre-loaded hf_model - causes conflicts
                    # The hf_model is already on the correct device from load_kwargs_hf above
                    
                    model = CustomHookedTransformer.from_pretrained(
                        model_id,
                        hf_model=hf_model,
                        **load_kwargs               
                    )
                except Exception as e:
                    logger.error(f"Direct loading failed: {e}")
                    raise ValueError(f"gpt-oss-20b architecture (MoE, GQA) not compatible with transformer_lens activation patching")
            else:
                # Use device_map OR device, not both (avoids tensor placement conflicts)
                load_kwargs = {
                    "tokenizer": tokenizer,
                    "n_devices": self.model_config.n_devices,
                    "torch_dtype": torch.bfloat16,
                    "fold_ln": self.model_config.fold_ln,
                    "center_writing_weights": False,
                    "center_unembed": False,
                }
                
# HF token is already configured via setup_hf_authentication()
                
                # Add quantization if specified
                if self.model_config.load_in_8bit:
                    load_kwargs["load_in_8bit"] = True
                    logger.info("Loading model with 8-bit quantization")
                elif self.model_config.load_in_4bit:
                    load_kwargs["load_in_4bit"] = True
                    logger.info("Loading model with 4-bit quantization")
                
                # For single device, use device instead of device_map to avoid fragmentation
                if self.model_config.device_map == "cuda" and self.model_config.n_devices == 1:
                    load_kwargs["device"] = "cuda"
                    load_kwargs["n_devices"] = 1
                    logger.info(f"using single device: cuda, forcing n_devices=1")
                elif self.model_config.device_map != "cpu":
                    load_kwargs["device_map"] = self.model_config.device_map
                    logger.info(f"using device_map: {self.model_config.device_map}")
                else:
                    load_kwargs["device"] = self.model_config.device
                    load_kwargs["n_devices"] = 1  # Force single device
                    logger.info(f"using single device: {self.model_config.device}, forcing n_devices=1")
                
                logger.info(f"load_kwargs: {load_kwargs}")
                try:
                    model = CustomHookedTransformer.from_pretrained(
                        model_id,
                        **load_kwargs               
                    )
                except Exception as e:
                    logger.error(f"CustomHookedTransformer.from_pretrained failed: {e}")
                    logger.error(f"load_kwargs were: {load_kwargs}")
                    raise 
            
            if self.model_config.ungroup_grouped_query_attention:
                logger.info("ungroup the keys/values which were grouped in the grouped query attention")
                model.set_ungroup_grouped_query_attention(True)
                logger.info(f"set use_past_kv_cache to False to avoid the error occurring in the generation with ungrouped keys/values")
            
            # Set model to eval mode for inference speed
            model.eval()
            logger.info("Model set to eval mode for inference")
                
            return model
                
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error("Failed to load the model. Please check the model id")
            raise
    
    def _setup_model_embeddings(self, model, tokenizer):
        """Resize model embeddings to match tokenizer if needed."""
        n_model = model.cfg.d_vocab
        n_tok = len(tokenizer)
        
        if n_tok > n_model:
            model.resize_token_embeddings(n_tok)
            logger.info(f"grow token embeddings: {n_model} -> {n_tok}")
        else:
            logger.info(f"no change {n_model} = {n_tok}")
    
    def _setup_special_tokens(self, model, tokenizer):
        """Setup EOS tokens and A/B tokens for evaluation."""
        # Setup EOS tokens
        eos_token_id = tokenizer.eos_token_id 
        vocab = tokenizer.get_vocab()
        
        # Skip newline eos tokens in dual-answer mode since schema contains newlines
        if not (self.prompt_config.use_template_system and self.prompt_config.ask_world):
            eos_token = self.generation_config.eos_token or "\n"
            add_eos_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(eos_token)[-1])
            logger.info(f"eos_token: {eos_token}, add_eos_token: {add_eos_token}")
            
            # FIXED: Only add the exact newline token, not all tokens starting with it
            try:
                exact_newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
                add_eos_token_id = [exact_newline_id]
                logger.info(f"DEBUG: Added exact newline token ID: {exact_newline_id}")
            except:
                add_eos_token_id = []
                logger.warning("Could not find exact newline token")
            
            if type(eos_token_id) != list:
                logger.info(f"eos_token_id: {eos_token_id}, eos: {tokenizer.convert_ids_to_tokens(eos_token_id)}")
                eos_token_id = [eos_token_id]
            eos_token_id.extend(add_eos_token_id)
        else:
            logger.info("skipping newline eos tokens for dual-answer mode")
            if type(eos_token_id) != list:
                logger.info(f"eos_token_id: {eos_token_id}, eos: {tokenizer.convert_ids_to_tokens(eos_token_id)}")
                eos_token_id = [eos_token_id]
        
        # FIXED: Don't add </loc> as EOS for behavioral TOM - it conflicts with <loc> generation
        # The model needs to be able to generate <loc>content</loc> freely
        
        if self.prompt_config.use_template_system and not self.prompt_config.ask_world:
            try:
                # Encode the FULL closing tag as a sequence
                loc_close_tokens = tokenizer.encode("</loc>", add_special_tokens=False)
                # Only use the last token if it uniquely represents </loc>
                # Otherwise skip it to avoid premature stopping
                if len(loc_close_tokens) == 1:
                    eos_token_id.append(loc_close_tokens[0])
                    logger.info(f"added </loc> as eos token: {loc_close_tokens[0]}")
                else:
                    logger.info(f"skipping </loc> as eos (multi-token: {loc_close_tokens})")
            except Exception as e:
                logger.warning(f"Error encoding </loc>: {e}")
                
        if self.prompt_config.use_template_system:
            logger.info("skipping </loc> eos token for behavioral TOM prompts to allow <loc> generation")
        
        # Setup A/B tokens for forced choice
        tokA = tokenizer.tokenize(" A")
        tokB = tokenizer.tokenize(" B")
        if len(tokA) == 1 and len(tokB) == 1:
            A_tok_id = tokenizer.convert_tokens_to_ids(tokA[0])
            B_tok_id = tokenizer.convert_tokens_to_ids(tokB[0])
            logger.info(f"using single-token answers: {tokA[0]} / {tokB[0]}")
        else:
            logger.info("adding <A>/<B> as special tokens for forced-choice answers")
            tokenizer.add_special_tokens({'additional_special_tokens': ['<A>', '<B>']})
            logger.info(f"resizing embeddings from {model.cfg.d_vocab} to {len(tokenizer)} for special tokens")
            model.resize_token_embeddings(len(tokenizer))
            A_tok_id = tokenizer.convert_tokens_to_ids('<A>')
            B_tok_id = tokenizer.convert_tokens_to_ids('<B>')
            logger.info(f"<A> token id: {A_tok_id}, <B> token id: {B_tok_id}")
            
        return eos_token_id, A_tok_id, B_tok_id