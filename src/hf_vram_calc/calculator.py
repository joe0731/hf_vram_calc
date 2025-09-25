"""
VRAM calculator for different data types and scenarios.

Memory Estimation Formulas:
==========================

1. Model Parameters Calculation:
   - TRANSFORMERS METHOD (Recommended): Use model.parameters() to get exact parameter count
   - MATHEMATICAL METHOD (Fallback): Mathematical estimation for unsupported architectures
   - Embedding: vocab_size × hidden_size
   - Attention: num_attention_heads × hidden_size × hidden_size × 4 (Q, K, V, O projections)
   - For GQA/MQA: Q = num_attention_heads × hidden_size², KV = num_key_value_heads × hidden_size² × 2
   - FFN: hidden_size × intermediate_size × 2 (up and down projections)
   - Layer Norm: hidden_size × 2 (weight and bias per layer norm)
   - Per Layer: (attention_params + ffn_params + ln_params)
   - Total: embedding + (per_layer × num_layers) + output_projection + final_ln

2. Memory Requirements:
   - Model Memory: num_parameters × bytes_per_dtype
   - KV Cache Memory: 2 × batch_size × sequence_length × hidden_size × num_layers × precision_bytes ÷ 1,073,741,824
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

import os
import shutil
import tempfile
import uuid
import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
from dataclasses import dataclass
from typing import Dict

from .config import ConfigManager
from .models import ModelConfig


@dataclass
class LlmodelMemoryResult:
    """Store all calculated memory values for a model and data type"""
    dtype: str
    num_params: int
    batch_size: int
    sequence_length: int
    lora_rank: int
    base_memory: float  # Model weights only
    kv_cache_memory: float
    inference_memory: float  # Model weights + overhead
    training_memory: float
    lora_memory: float

    def __init__(self, dtype: str, batch_size: int = 1, sequence_length: int = 2048, lora_rank: int = 64):
        """Initialize LlmodelMemoryResult with basic parameters"""
        self.dtype = dtype
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.lora_rank = lora_rank
        # Initialize calculated values with defaults
        self.num_params = 0
        self.base_memory = 0.0
        self.kv_cache_memory = 0.0
        self.inference_memory = 0.0
        self.training_memory = 0.0
        self.lora_memory = 0.0

    def calculate_all(self, config: ModelConfig, num_params: int, vram_calc: 'VRAMCalculator'):
        """Calculate all memory values using instance parameters"""
        # Update num_params
        self.num_params = num_params
        # Calculate base model memory
        self.base_memory = vram_calc.calculate_model_memory(num_params, self.dtype)
        # Calculate KV cache memory using instance parameters
        self.kv_cache_memory = vram_calc.calculate_kv_cache_memory(
            config, self.dtype, self.batch_size, self.sequence_length
        )
        # Calculate inference memory (model + overhead)
        self.inference_memory = vram_calc.calculate_inference_memory(self.base_memory)
        # Calculate training memory
        self.training_memory = vram_calc.calculate_training_memory(self.base_memory)
        # Calculate LoRA memory using instance parameters
        self.lora_memory = vram_calc.calculate_lora_memory(self.base_memory, num_params, self.lora_rank)


class ParameterCalculator:
    """Calculate model parameters for different architectures"""
    
    @staticmethod
    def calculate_parameters_by_transformers(config: ModelConfig) -> int:
        """
        Calculate exact parameter count using model.parameters() method.
        This is the most accurate method as it uses the actual model structure.
        Uses the global cache directory from ConfigParser.
        """
        try:
            test_config = config.test_config

            if test_config is None:
                raise ValueError("test_config is None - config object not properly stored")

            # Get the actual torch_dtype from config, fallback to bfloat16 if not specified
            torch_dtype = getattr(config, 'torch_dtype', torch.bfloat16)
            if isinstance(torch_dtype, str):
                # Convert string dtype to torch dtype
                torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)

            # Create empty model to get exact parameter count
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(test_config, dtype=torch_dtype)

            # Count total parameters
            total_params = 0
            for param in model.parameters():
                total_params += param.numel()

            return total_params

        except Exception as e:
            # If accurate method fails, fall back to mathematical estimation
            print(f"Warning: Accurate parameter calculation failed for {config.model_name}: {e}")
            print("Falling back to mathematical estimation...")
            return None

    @staticmethod
    def calculate_transformer_params(config: ModelConfig) -> int:
        """Calculate parameters for transformer-based models"""
        # Try accurate method first
        transformers_params = ParameterCalculator.calculate_parameters_by_transformers(config)
        if transformers_params is not None:
            return transformers_params
        else:
            raise ValueError(f"Failed to calculate parameters for {config.model_name}")


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
    
    def calculate_kv_cache_memory(self, config: ModelConfig, dtype: str, batch_size: int = 1,
                                 sequence_length: int = 2048) -> float:
        """
        Calculate KV Cache memory requirements in GB.
        Formula: KV_Cache (GB) = 2 × Batch_Size × Sequence_Length × Head_Dim × Num_KV_Heads × Num_Layers × Precision ÷ 1,073,741,824
        
        Supports MHA/MQA/GQA via model configuration:
        - MHA: Num_KV_Heads = Num_Query_Heads (i.e., config.num_key_value_heads is None)
        - MQA: Num_KV_Heads = 1 (i.e., config.num_key_value_heads == 1)
        - GQA: Num_KV_Heads = Num_Query_Heads / Num_Groups (as provided in config.num_key_value_heads)
        
        Implementation details:
        - Num_KV_Heads is taken from config.num_key_value_heads if present; otherwise falls back to config.num_attention_heads
        - Head_Dim = hidden_size // num_attention_heads
        Args:
            config: Model configuration
            dtype: Data type for the model (KV cache uses appropriate precision, not necessarily same as model)
            batch_size: Batch size for inference
            sequence_length: Sequence length for KV cache
        Returns:
            KV Cache memory in GB
        """
        # KV Cache typically uses FP16/BF16 precision even for quantized models
        # INT4/INT8 quantized models still use FP16/BF16 for KV cache
        if dtype in ['int4', 'int8']:
            # Use FP16 precision for KV cache (2 bytes) for quantized models
            precision_bytes = 2
        else:
            # Use the same precision as the model for non-quantized models
            precision_bytes = int(self.get_dtype_size(dtype))

        # Derive the number of KV heads from config; fallback to query heads when unspecified
        num_kv_heads = config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads

        # Head dimension derived from hidden size and number of attention heads
        if config.num_attention_heads and config.num_attention_heads > 0:
            head_dim = config.hidden_size // config.num_attention_heads
        else:
            head_dim = config.hidden_size

        # General KV cache formula accounting for GQA/MQA/MHA:
        # 2 (K and V) × B × S × head_dim × num_kv_heads × L × precision_bytes
        kv_cache_bytes = (
            2
            * batch_size
            * sequence_length
            * head_dim
            * num_kv_heads
            * config.num_layers
            * precision_bytes
        )
        # Convert to GB (divide by 1,073,741,824)
        kv_cache_gb = kv_cache_bytes / (1024 ** 3)
        return kv_cache_gb

    def calculate_inference_memory(self, model_memory_gb: float, 
                                 kv_cache_factor: float = None) -> float:
        """Calculate inference memory requirements (model weights only)"""
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
