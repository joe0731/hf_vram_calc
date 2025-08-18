"""
VRAM calculator for different data types and scenarios.

Memory Estimation Formulas:
==========================

1. Model Parameters Calculation:
   - Embedding: vocab_size × hidden_size
   - Attention: num_attention_heads × hidden_size × hidden_size × 4 (Q, K, V, O projections)
   - For GQA/MQA: Q = num_attention_heads × hidden_size², KV = num_key_value_heads × hidden_size² × 2
   - FFN: hidden_size × intermediate_size × 2 (up and down projections)
   - Layer Norm: hidden_size × 2 (weight and bias per layer norm)
   - Per Layer: (attention_params + ffn_params + ln_params)
   - Total: embedding + (per_layer × num_layers) + output_projection + final_ln

2. Memory Requirements:
   - Model Memory: num_parameters × bytes_per_dtype
   - Inference Memory: model_memory × 1.2 (includes KV cache and overhead)
   - Training Memory: model_memory × 4 × 1.3 (weights + gradients + optimizer_states × overhead)
   - LoRA Memory: model_memory + (lora_params × 4 × 2) × 1.2 (trainable params with optimizer)

3. Activation Memory (per token):
   - Hidden States: batch_size × seq_length × hidden_size × bytes_per_activation
   - Attention Scores: batch_size × num_heads × seq_length × (hidden_size/num_heads) × bytes_per_activation
   - FFN Intermediate: batch_size × seq_length × intermediate_size × bytes_per_activation
   - Total: sum of above × active_layers_factor

4. LoRA Parameters Estimation:
   - Target Parameters: total_params × target_modules_ratio
   - LoRA Parameters: target_params × (2 × rank / original_dim)
   - Memory Overhead: lora_params × bytes_per_param × 4 (for gradients + optimizer)
"""

from .config import ConfigManager
from .models import ModelConfig


class ParameterCalculator:
    """Calculate model parameters for different architectures"""
    
    @staticmethod
    def calculate_embedding_params(vocab_size: int, hidden_size: int) -> int:
        """Calculate embedding layer parameters"""
        return vocab_size * hidden_size
    
    @staticmethod
    def calculate_attention_params(hidden_size: int, num_attention_heads: int, 
                                 num_key_value_heads: int = None) -> int:
        """Calculate attention layer parameters"""
        if num_key_value_heads is not None and num_key_value_heads != num_attention_heads:
            # grouped-query attention or multi-query attention
            q_params = num_attention_heads * hidden_size * hidden_size
            kv_params = num_key_value_heads * hidden_size * hidden_size * 2
            output_params = hidden_size * hidden_size
            return q_params + kv_params + output_params
        else:
            # standard multi-head attention (Q, K, V, O projections)
            return 4 * hidden_size * hidden_size
    
    @staticmethod
    def calculate_ffn_params(hidden_size: int, intermediate_size: int = None) -> int:
        """Calculate feed-forward network parameters"""
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        # up projection + down projection
        return 2 * hidden_size * intermediate_size
    
    @staticmethod
    def calculate_layer_norm_params(hidden_size: int) -> int:
        """Calculate layer normalization parameters"""
        # weight and bias
        return 2 * hidden_size
    
    @staticmethod
    def calculate_transformer_params(config: ModelConfig) -> int:
        """Calculate parameters for transformer-based models"""
        # embedding parameters
        embedding_params = ParameterCalculator.calculate_embedding_params(
            config.vocab_size, config.hidden_size
        )
        
        # attention parameters
        attention_params = ParameterCalculator.calculate_attention_params(
            config.hidden_size, config.num_attention_heads, config.num_key_value_heads
        )
        
        # feed-forward network parameters
        ffn_params = ParameterCalculator.calculate_ffn_params(
            config.hidden_size, config.intermediate_size
        )
        
        # layer normalization parameters (usually 2 per layer: pre-attention and pre-ffn)
        ln_params_per_layer = ParameterCalculator.calculate_layer_norm_params(config.hidden_size) * 2
        
        # total parameters per layer
        per_layer_params = attention_params + ffn_params + ln_params_per_layer
        
        # all layers
        all_layers_params = per_layer_params * config.num_layers
        
        # output projection (usually tied with embeddings, but count separately for accuracy)
        output_params = config.vocab_size * config.hidden_size
        
        # final layer normalization
        final_ln_params = ParameterCalculator.calculate_layer_norm_params(config.hidden_size)
        
        total_params = embedding_params + all_layers_params + output_params + final_ln_params
        
        return total_params


class VRAMCalculator:
    """Calculate VRAM requirements for different data types and scenarios"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.dtype_sizes = config_manager.get_data_types()
        
        # memory overhead factors
        self.inference_overhead_factor = 1.2
        self.training_base_factor = 4.0  # weights + gradients + optimizer states
        self.training_overhead_factor = 1.3
        self.lora_overhead_factor = 1.2
        self.activation_bytes = 2  # assume fp16/bf16 for activations
        self.active_layers_factor = 4  # conservative estimate for activation storage
    
    def get_dtype_size(self, dtype: str) -> float:
        """Get bytes per parameter for given data type"""
        if dtype not in self.dtype_sizes:
            available_types = list(self.dtype_sizes.keys())
            raise ValueError(f"unsupported data type: {dtype}. Available types: {available_types}")
        
        dtype_info = self.dtype_sizes[dtype]
        if isinstance(dtype_info, dict):
            return dtype_info.get('bytes_per_param', dtype_info)
        return dtype_info
    
    def calculate_model_memory(self, num_params: int, dtype: str) -> float:
        """Calculate model weights memory in GB"""
        bytes_per_param = self.get_dtype_size(dtype)
        total_bytes = num_params * bytes_per_param
        return total_bytes / (1024 ** 3)  # convert to GB
    
    def calculate_inference_memory(self, model_memory_gb: float, 
                                 kv_cache_factor: float = None) -> float:
        """Calculate inference memory requirements"""
        if kv_cache_factor is None:
            kv_cache_factor = self.inference_overhead_factor
        
        return model_memory_gb * kv_cache_factor
    
    def calculate_training_memory(self, model_memory_gb: float, 
                                optimizer_factor: float = None,
                                overhead_factor: float = None) -> float:
        """Calculate training memory requirements"""
        if optimizer_factor is None:
            optimizer_factor = self.training_base_factor
        if overhead_factor is None:
            overhead_factor = self.training_overhead_factor
        
        return model_memory_gb * optimizer_factor * overhead_factor
    
    def calculate_lora_memory(self, model_memory_gb: float, num_params: int, 
                            lora_rank: int = 64, target_modules_ratio: float = 0.25,
                            original_dim: int = 4096) -> float:
        """Calculate LoRA fine-tuning memory requirements"""
        # estimate trainable parameters for LoRA
        target_params = num_params * target_modules_ratio
        lora_params = target_params * (2 * lora_rank / original_dim)
        lora_params_billions = lora_params / 1e9
        
        # LoRA memory overhead (trainable params with gradients and optimizer states)
        lora_overhead_gb = lora_params_billions * self.get_dtype_size('fp16') * self.training_base_factor
        
        total_memory = (model_memory_gb + lora_overhead_gb) * self.lora_overhead_factor
        return total_memory
    
    def estimate_activation_memory(self, config: ModelConfig, batch_size: int = 1, 
                                 sequence_length: int = 2048,
                                 activation_bytes: int = None) -> float:
        """Estimate activation memory in GB"""
        if activation_bytes is None:
            activation_bytes = self.activation_bytes
        
        # hidden states activation
        hidden_states_size = batch_size * sequence_length * config.hidden_size * activation_bytes
        
        # attention scores activation
        head_dim = config.hidden_size // config.num_attention_heads
        attention_scores_size = (batch_size * config.num_attention_heads * 
                               sequence_length * head_dim * activation_bytes)
        
        # ffn intermediate activation
        intermediate_size = config.intermediate_size
        if intermediate_size is None:
            intermediate_size = 4 * config.hidden_size
        ffn_activation_size = batch_size * sequence_length * intermediate_size * activation_bytes
        
        # per layer activation size
        per_layer_size = hidden_states_size + attention_scores_size + ffn_activation_size
        
        # total activation memory (not all layers store activations simultaneously)
        active_layers = min(config.num_layers, self.active_layers_factor)
        total_activation_size = per_layer_size * active_layers
        
        return total_activation_size / (1024 ** 3)  # convert to GB
    
    def calculate_gradient_memory(self, model_memory_gb: float) -> float:
        """Calculate gradient memory requirements"""
        return model_memory_gb  # gradients have same size as model weights
    
    def calculate_optimizer_memory(self, model_memory_gb: float, optimizer_type: str = 'adam') -> float:
        """Calculate optimizer state memory requirements"""
        optimizer_factors = {
            'sgd': 0,  # no additional memory for SGD
            'momentum': 1,  # momentum buffer
            'adam': 2,  # first and second moment estimates
            'adamw': 2,  # same as adam
            'rmsprop': 1,  # squared gradient average
        }
        
        factor = optimizer_factors.get(optimizer_type.lower(), 2)  # default to adam
        return model_memory_gb * factor
