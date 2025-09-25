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
    def _get_token() -> Optional[str]:
        """Get Hugging Face token from environment variables or cache"""
        # Try environment variables first
        token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        if token:
            return token

        # Try to read from HF CLI cache
        try:
            import pathlib
            token_file = pathlib.Path.home() / '.cache' / 'huggingface' / 'token'
            if token_file.exists():
                token = token_file.read_text().strip()
                if token:
                    return token
        except:
            pass

        return None

    @staticmethod
    def _request_token_interactively() -> Optional[str]:
        """Request Hugging Face token from user interactively"""
        try:
            print("\n🔐 This model requires Hugging Face authentication.")
            print("You can get your token from: https://huggingface.co/settings/tokens")
            token = input("Enter your Hugging Face token (or press Enter to skip): ").strip()
            if not token:
                return None
            # Validate token format (basic check)
            if not token.startswith('hf_') and len(token) < 20:
                print("Warning: Token doesn't look like a valid Hugging Face token.")
                print("Valid tokens usually start with 'hf_' and are longer.")
                retry = input("Continue anyway? (y/N): ").strip().lower()
                if retry not in ['y', 'yes']:
                    return None
            return token
        except KeyboardInterrupt:
            print("\nToken input cancelled.")
            return None
        except Exception as e:
            print(f"Error getting token input: {e}")
            return None

    @staticmethod
    def safe_get(obj, *keys):
        """Safely get attribute from config object or dict"""
        for key in keys:
            if hasattr(obj, key):
                return getattr(obj, key)
            elif hasattr(obj, 'get') and callable(getattr(obj, 'get')):
                return obj.get(key)
        return None

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
            token = ConfigParser._get_token()

            if token:
                headers['Authorization'] = f'Bearer {token}'

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Save config to cache
            with open(cached_config_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2)
            return cached_config_path

        except requests.RequestException as e:
            # If 403 Forbidden, try to get token from user
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 403:
                token = ConfigParser._request_token_interactively()
                if token:
                    headers['Authorization'] = f'Bearer {token}'
                    try:
                        response = requests.get(url, headers=headers, timeout=10)
                        response.raise_for_status()
                        # Save config to cache
                        with open(cached_config_path, 'w', encoding='utf-8') as f:
                            json.dump(response.json(), f, indent=2)
                        return cached_config_path
                    except requests.RequestException as retry_e:
                        error_msg = (
                            f"failed to fetch config for model '{model_name}' even with token: {retry_e}. "
                            "Please check your token or try using --local-config option"
                        )
                        raise RuntimeError(error_msg)
                else:
                    error_msg = (
                        f"failed to fetch config for model '{model_name}': {e}. "
                        "This model may require authentication. Please use --local-config option or run 'huggingface-cli login'"
                    )
                    raise RuntimeError(error_msg)
            else:
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
                # This model config is MOE, using text_config
            else:
                text_config = cfg
                # This model config is a causal model, using root config

            hidden_size = ConfigParser.safe_get(text_config, "hidden_size", "n_embd", "d_model")
            num_layers = ConfigParser.safe_get(text_config, "num_hidden_layers", "num_layers", "n_layer", "n_layers")
            num_attention_heads = ConfigParser.safe_get(text_config, "num_attention_heads", "n_head", "num_heads")
            intermediate_size = ConfigParser.safe_get(text_config, "intermediate_size", "n_inner", "d_ff")

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
            torch_dtype = ConfigParser.safe_get(text_config, "torch_dtype") or ConfigParser.safe_get(cfg, "torch_dtype")
            recommended_dtype = ConfigParser.map_torch_dtype_to_our_dtype(torch_dtype, model_name)

            return ModelConfig(
                model_name=model_name,
                model_type=ConfigParser.safe_get(text_config, "model_type"),
                vocab_size=ConfigParser.safe_get(text_config, "vocab_size"),
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=ConfigParser.safe_get(text_config, "num_key_value_heads"),
                max_position_embeddings=ConfigParser.safe_get(text_config, "max_position_embeddings", "n_positions"),
                rope_theta=ConfigParser.safe_get(text_config, "rope_theta"),
                torch_dtype=torch_dtype,
                recommended_dtype=recommended_dtype,
                test_config=text_config # to save the original test config
            )
        except KeyError as e:
            raise ValueError(f"missing required config field: {e}")
