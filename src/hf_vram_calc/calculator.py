"""
VRAM calculator for different data types and scenarios.
"""

from .config import ConfigManager
from .models import ModelConfig


class ParameterCalculator:
    """Calculate model parameters for different architectures"""
    
    @staticmethod
    def calculate_transformer_params(config: ModelConfig) -> int:
        """Calculate parameters for transformer-based models"""
        # Embedding parameters
        embedding_params = config.vocab_size * config.hidden_size
        
        # Layer parameters
        layer_params = 0
        
        # Attention parameters
        if config.num_key_value_heads:
            # Multi-query or grouped-query attention
            q_params = config.num_attention_heads * config.hidden_size * config.hidden_size
            kv_params = config.num_key_value_heads * config.hidden_size * config.hidden_size * 2
            attention_params = q_params + kv_params
        else:
            # Standard multi-head attention
            attention_params = 4 * config.hidden_size * config.hidden_size
        
        # Feed-forward network parameters
        if config.intermediate_size:
            ffn_params = 2 * config.hidden_size * config.intermediate_size
        else:
            # Default to 4x hidden size if not specified
            ffn_params = 2 * config.hidden_size * (4 * config.hidden_size)
        
        # Layer normalization parameters
        ln_params = 2 * config.hidden_size
        
        layer_params = (attention_params + ffn_params + ln_params) * config.num_layers
        
        # Output layer parameters (usually tied with embeddings, but count separately)
        output_params = config.vocab_size * config.hidden_size
        
        # Final layer norm
        final_ln_params = config.hidden_size
        
        total_params = embedding_params + layer_params + output_params + final_ln_params
        
        return total_params


class VRAMCalculator:
    """Calculate VRAM requirements for different data types and scenarios"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.dtype_sizes = config_manager.get_data_types()
    
    def calculate_model_memory(self, num_params: int, dtype: str) -> float:
        """Calculate model weights memory in GB"""
        if dtype not in self.dtype_sizes:
            available_types = list(self.dtype_sizes.keys())
            raise ValueError(f"unsupported data type: {dtype}. Available types: {available_types}")
        
        bytes_per_param = self.dtype_sizes[dtype]
        total_bytes = num_params * bytes_per_param
        return total_bytes / (1024 ** 3)  # convert to GB
    
    @staticmethod
    def calculate_inference_memory(model_memory_gb: float) -> float:
        """Calculate inference memory requirements"""
        # inference memory = model size × 1.2 (includes KV cache and other overhead)
        return model_memory_gb * 1.2
    
    @staticmethod
    def calculate_training_memory(model_memory_gb: float) -> float:
        """Calculate training memory requirements with Adam optimizer"""
        # training memory = model size × 4 × 1.3
        # 4x = model weights (1x) + gradients (1x) + optimizer states (2x for Adam)
        # 1.3x = 30% overhead for other activations and buffers
        return model_memory_gb * 4 * 1.3
    
    @staticmethod
    def calculate_lora_memory(model_memory_gb: float, num_params: int, 
                            lora_rank: int = 64, target_modules_ratio: float = 0.25) -> float:
        """Calculate LoRA fine-tuning memory requirements"""
        # Estimate trainable parameters for LoRA
        # typically LoRA targets 25% of model parameters (attention layers)
        # LoRA params ≈ 2 × rank × (target_params / attention_head_dim) × target_ratio
        trainable_params = num_params * target_modules_ratio * (lora_rank * 2 / 4096)  # rough estimation
        trainable_params_billions = trainable_params / 1e9
        
        # LoRA memory = (model size + trainable_params_billions × 16/8 × 4) × 1.2
        lora_overhead_gb = trainable_params_billions * 2 * 4  # 16/8 = 2, multiply by 4 for gradients/optimizer
        return (model_memory_gb + lora_overhead_gb) * 1.2
    
    @staticmethod
    def estimate_activation_memory(config: ModelConfig, batch_size: int = 1, 
                                 sequence_length: int = 2048) -> float:
        """Estimate activation memory in GB (rough estimation)"""
        # Rough estimation for transformer activations
        hidden_states_size = batch_size * sequence_length * config.hidden_size * 2  # fp16/bf16
        attention_scores_size = batch_size * config.num_attention_heads * sequence_length * (config.hidden_size // config.num_attention_heads) * 2
        
        # Intermediate activations (FFN)
        ffn_intermediate_size = config.intermediate_size if config.intermediate_size else (4 * config.hidden_size)
        ffn_activation_size = batch_size * sequence_length * ffn_intermediate_size * 2
        
        # Per layer activation size
        per_layer_size = hidden_states_size + attention_scores_size + ffn_activation_size
        
        # Total activation size (considering only a few layers need to store activations at once)
        # during inference, not all layers store activations simultaneously
        total_activation_size = per_layer_size * min(config.num_layers, 4)  # more conservative estimate
        
        return total_activation_size / (1024 ** 3)  # convert to GB
