"""
Configuration parser for Hugging Face models.
"""

import json
import os
import shutil
import tempfile
import uuid
import requests
from pathlib import Path
from typing import Dict, Optional

from transformers import AutoConfig

from .models import ModelConfig


class ConfigParser:
    """Parse model configuration from Hugging Face"""
    
    # Global temporary cache directory
    _global_cache_dir: Optional[str] = None

    @classmethod
    def get_global_cache_dir(cls) -> str:
        """Get or create global temporary cache directory"""
        if cls._global_cache_dir is None:
            cls._global_cache_dir = f"/tmp/hf_vram_calc_cache_{uuid.uuid4().hex[:8]}"
            os.makedirs(cls._global_cache_dir, exist_ok=True)
        return cls._global_cache_dir

    @classmethod
    def cleanup_global_cache(cls):
        """Clean up global temporary cache directory"""
        if cls._global_cache_dir and os.path.exists(cls._global_cache_dir):
            try:
                shutil.rmtree(cls._global_cache_dir)
                cls._global_cache_dir = None
            except Exception as e:
                print(f"Warning: Failed to clean up cache directory {cls._global_cache_dir}: {e}")

    @staticmethod
    def extract_dtype_from_model_name(model_name: str) -> Optional[str]:
        """Extract data type from model name if present"""
        model_name_lower = model_name.lower()
        
        # Common patterns in model names for data types
        dtype_patterns = {
            'fp32': ['fp32', 'float32'],
            'fp16': ['fp16', 'float16', 'half'],
            'bf16': ['bf16', 'bfloat16', 'brain-float16', 'brainf16'],
            'fp8': ['fp8', 'float8'],
            'int8': ['int8', '8bit', 'w8a16'],
            'int4': ['int4', '4bit', 'w4a16', 'gptq', 'awq'],
            'nf4': ['nf4', 'bnb-4bit'],
            'awq_int4': ['awq-int4', 'awq_int4'],
            'gptq_int4': ['gptq-int4', 'gptq_int4'],
        }
        
        # Look for dtype patterns in model name
        for our_dtype, patterns in dtype_patterns.items():
            for pattern in patterns:
                if pattern in model_name_lower:
                    return our_dtype
        
        return None
    
    @staticmethod
    def map_torch_dtype_to_our_dtype(torch_dtype: Optional[str], model_name: str = "") -> str:
        """Map torch_dtype from config to our data type format with model name priority"""
        
        # Priority 1: Extract from model name
        if model_name:
            dtype_from_name = ConfigParser.extract_dtype_from_model_name(model_name)
            if dtype_from_name:
                return dtype_from_name
        
        # Priority 2: Use config torch_dtype
        if torch_dtype:
            # normalize the torch_dtype string
            torch_dtype_lower = str(torch_dtype).lower().strip()
            
            # mapping from torch dtype to our dtype format
            dtype_mapping = {
                "torch.float32": "fp32",
                "torch.float": "fp32", 
                "float32": "fp32",
                "float": "fp32",
                "torch.float16": "fp16",
                "float16": "fp16",
                "torch.bfloat16": "bf16", 
                "bfloat16": "bf16",
                "torch.float8": "fp8",
                "float8": "fp8",
                "torch.int8": "int8",
                "int8": "int8",
                "torch.int4": "int4",
                "int4": "int4",
            }
            
            mapped_dtype = dtype_mapping.get(torch_dtype_lower)
            if mapped_dtype:
                return mapped_dtype

        # Priority 3: Default to fp16
        return "fp16"

    @staticmethod
    def fetch_config(model_name: str, local_config_path: Optional[str] = None) -> str:
        """Fetch config.json and return the cached file path"""
        global_cache_dir = ConfigParser.get_global_cache_dir()

        # Use local config if provided
        if local_config_path:
            try:
                config_path = Path(local_config_path)
                if not config_path.exists():
                    raise FileNotFoundError(f"local config file not found: {local_config_path}")
                # Copy local config to global cache
                cached_config_path = os.path.join(global_cache_dir, f"{model_name.replace('/', '_')}_config.json")
                shutil.copy2(config_path, cached_config_path)
                return cached_config_path

            except (json.JSONDecodeError, FileNotFoundError) as e:
                raise RuntimeError(f"failed to load local config from '{local_config_path}': {e}.\n"
                                 f"please check if your config json file format is correct and complete")

        # Check if config already exists in cache
        cached_config_path = os.path.join(global_cache_dir, f"{model_name.replace('/', '_')}_config.json")
        if os.path.exists(cached_config_path):
            return cached_config_path

        # Fetch from Hugging Face if no local config specified
        try:
            url = f"https://huggingface.co/{model_name}/raw/main/config.json"

            # Add authentication headers if token is available
            headers = {}
            token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
            if not token:
                # Try to read from HF CLI cache
                try:
                    import pathlib
                    token_file = pathlib.Path.home() / '.cache' / 'huggingface' / 'token'
                    if token_file.exists():
                        token = token_file.read_text().strip()
                except:
                    pass

            if token:
                headers['Authorization'] = f'Bearer {token}'

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Save config to cache
            with open(cached_config_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2)
            return cached_config_path

        except requests.RequestException as e:
            error_msg = (
                f"failed to fetch config for model '{model_name}': {e}. "
                "Please check network connection or try using --local-config option"
            )
            raise RuntimeError(error_msg)

    @staticmethod
    def parse_config(config_path: str, model_name: str) -> ModelConfig:
        """Parse config file into ModelConfig"""
        try:
            # Load config from AutoConfig
            cfg = AutoConfig.from_pretrained(config_path, local_files_only=True)
            if hasattr(cfg, 'text_config'):
                text_config = cfg.text_config
                print(f"This model config is MOE, using text_config for {model_name}")
            else:
                text_config = cfg
                print(f"This model config is a causal model, using root config for {model_name}")
            # Check if this is a multimodal model with text_config
            hidden_size = (text_config.get("hidden_size") or
                          text_config.get("n_embd") or
                          text_config.get("d_model"))

            num_layers = (text_config.get("num_hidden_layers") or
                         text_config.get("num_layers") or
                         text_config.get("n_layer") or
                         text_config.get("n_layers"))

            num_attention_heads = (text_config.get("num_attention_heads") or
                         text_config.get("n_head") or
                         text_config.get("num_heads"))

            # Handle different field names for different model architectures
            hidden_size = (text_config.get("hidden_size") or
                          text_config.get("n_embd") or
                          text_config.get("d_model"))

            num_layers = (text_config.get("num_hidden_layers") or
                         text_config.get("num_layers") or
                         text_config.get("n_layer") or
                         text_config.get("n_layers"))

            num_attention_heads = (text_config.get("num_attention_heads") or
                                 text_config.get("n_head") or
                                 text_config.get("num_heads"))
            
            intermediate_size = (text_config.get("intermediate_size") or
                               text_config.get("n_inner") or
                               text_config.get("d_ff"))

            if not all([hidden_size, num_layers, num_attention_heads]):
                missing_fields = []
                if not hidden_size:
                    missing_fields.append("hidden_size/n_embd/d_model")
                if not num_layers:
                    missing_fields.append("num_hidden_layers/num_layers/n_layer")
                if not num_attention_heads:
                    missing_fields.append("num_attention_heads/n_head")
                raise ValueError(f"missing required config fields: {missing_fields}")

            # Extract torch_dtype and determine recommended data type
            # For multimodal models, prefer text_config torch_dtype, fallback to root
            torch_dtype = text_config.get("torch_dtype") or cfg.get("torch_dtype")
            recommended_dtype = ConfigParser.map_torch_dtype_to_our_dtype(torch_dtype, model_name)
            
            return ModelConfig(
                model_name=model_name,
                model_type=text_config.get("model_type"),
                vocab_size=text_config["vocab_size"],
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=text_config.get("num_key_value_heads"),
                max_position_embeddings=text_config.get("max_position_embeddings", text_config.get("n_positions")),
                rope_theta=text_config.get("rope_theta"),
                torch_dtype=torch_dtype,
                recommended_dtype=recommended_dtype,
                test_config=text_config # to save the original test config
            )
        except KeyError as e:
            raise ValueError(f"missing required config field: {e}")
