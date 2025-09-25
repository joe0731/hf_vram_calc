"""
Data models for HF VRAM Calculator.
"""

from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ModelConfig:
    """Model configuration data structure"""
    model_name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rope_theta: Optional[float] = None
    model_type: str = "unknown"
    torch_dtype: Optional[str] = None
    recommended_dtype: Optional[str] = None
    test_config: Optional[Any] = None  # Stores the original config object from AutoConfig
