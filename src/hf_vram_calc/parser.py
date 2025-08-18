"""
Configuration parser for Hugging Face models.
"""

import requests
from typing import Dict, Optional

from .models import ModelConfig


class ConfigParser:
    """Parse model configuration from Hugging Face"""
    
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
    def fetch_config(model_name: str) -> Dict:
        """Fetch config.json from Hugging Face"""
        try:
            url = f"https://huggingface.co/{model_name}/raw/main/config.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"failed to fetch config for {model_name}: {e}")
    
    @staticmethod
    def parse_config(config_data: Dict, model_name: str) -> ModelConfig:
        """Parse raw config data into ModelConfig"""
        try:
            # Handle different field names for different model architectures
            hidden_size = (config_data.get("hidden_size") or 
                          config_data.get("n_embd") or 
                          config_data.get("d_model"))
            
            num_layers = (config_data.get("num_hidden_layers") or 
                         config_data.get("num_layers") or 
                         config_data.get("n_layer") or 
                         config_data.get("n_layers"))
            
            num_attention_heads = (config_data.get("num_attention_heads") or 
                                 config_data.get("n_head") or 
                                 config_data.get("num_heads"))
            
            intermediate_size = (config_data.get("intermediate_size") or 
                               config_data.get("n_inner") or 
                               config_data.get("d_ff"))
            
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
            torch_dtype = config_data.get("torch_dtype")
            recommended_dtype = ConfigParser.map_torch_dtype_to_our_dtype(torch_dtype, model_name)
            
            return ModelConfig(
                model_name=model_name,
                model_type=config_data.get("model_type", "unknown"),
                vocab_size=config_data["vocab_size"],
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=config_data.get("num_key_value_heads"),
                max_position_embeddings=config_data.get("max_position_embeddings", config_data.get("n_positions")),
                rope_theta=config_data.get("rope_theta"),
                torch_dtype=torch_dtype,
                recommended_dtype=recommended_dtype
            )
        except KeyError as e:
            raise ValueError(f"missing required config field: {e}")
