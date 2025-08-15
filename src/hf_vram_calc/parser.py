"""
Configuration parser for Hugging Face models.
"""

import requests
from typing import Dict

from .models import ModelConfig


class ConfigParser:
    """Parse model configuration from Hugging Face"""
    
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
                rope_theta=config_data.get("rope_theta")
            )
        except KeyError as e:
            raise ValueError(f"missing required config field: {e}")
