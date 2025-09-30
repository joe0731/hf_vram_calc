#!/usr/bin/env python3
"""
Test script for hierarchical YAML configuration functionality
"""

import sys
import os
import tempfile
import yaml
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_hierarchical_yaml():
    """Test hierarchical YAML configuration parsing"""
    print("Testing hierarchical YAML configuration...")
    
    # Create a test YAML file with hierarchical structure
    test_config = {
        'model': 'test-model',
        'build_config': {
            'max_batch_size': 32,
            'max_seq_len': 4096,
            'max_num_tokens': 8192
        },
        'lora_config': {
            'max_lora_rank': 128,
            'lora_dir': '/path/to/lora'
        },
        'kv_cache_config': {
            'dtype': 'fp8',
            'mamba_ssm_cache_dtype': 'fp16'
        },
        'quant_config': {
            'quant_algo': 'fp8',
            'kv_cache_quant_algo': 'fp8'
        },
        'performance_options': {
            'cuda_graphs': True,
            'multi_block_mode': True
        },
        'decoding_config': {
            'medusa_choices': [[0, 1], [0, 2], [1, 2]]
        },
        'enable_chunked_prefill': True,
        'log_level': 'verbose'
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        yaml_path = f.name
    
    try:
        # Import the functions we need
        from hf_vram_calc.cli import load_yaml_config, apply_yaml_overrides
        
        # Test loading
        loaded_config = load_yaml_config(yaml_path)
        print(f"✅ YAML loaded successfully")
        print(f"   Model: {loaded_config.get('model')}")
        print(f"   Build config: {loaded_config.get('build_config')}")
        print(f"   LoRA config: {loaded_config.get('lora_config')}")
        
        # Test argument override
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default=None)
        parser.add_argument('--max_batch_size', type=int, default=1)
        parser.add_argument('--max_seq_len', type=int, default=2048)
        parser.add_argument('--lora_rank', type=int, default=64)
        parser.add_argument('--dtype', default=None)
        parser.add_argument('--log_level', default='info')
        
        args = parser.parse_args([])
        print(f"\nBefore override:")
        print(f"   model={args.model}, max_batch_size={args.max_batch_size}")
        print(f"   max_seq_len={args.max_seq_len}, lora_rank={args.lora_rank}")
        print(f"   dtype={args.dtype}, log_level={args.log_level}")
        
        args = apply_yaml_overrides(args, loaded_config)
        print(f"\nAfter override:")
        print(f"   model={args.model}, max_batch_size={args.max_batch_size}")
        print(f"   max_seq_len={args.max_seq_len}, lora_rank={args.lora_rank}")
        print(f"   dtype={args.dtype}, log_level={args.log_level}")
        
        # Verify overrides
        assert args.model == 'test-model'
        assert args.max_batch_size == 32
        assert args.max_seq_len == 4096  # Should use max_seq_len, not max_num_tokens
        assert args.lora_rank == 128
        assert args.dtype == 'fp8'  # Should come from kv_cache_config
        assert args.log_level == 'verbose'
        
        # Check that additional configs are stored
        assert hasattr(args, 'performance_options')
        assert hasattr(args, 'decoding_config')
        assert hasattr(args, 'enable_chunked_prefill')
        
        print("✅ All hierarchical YAML tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        os.unlink(yaml_path)

def test_quant_algo_mapping():
    """Test quantization algorithm to dtype mapping"""
    print("\nTesting quantization algorithm mapping...")
    
    test_config = {
        'quant_config': {
            'quant_algo': 'bf16'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        yaml_path = f.name
    
    try:
        from hf_vram_calc.cli import load_yaml_config, apply_yaml_overrides
        
        loaded_config = load_yaml_config(yaml_path)
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--dtype', default=None)
        
        args = parser.parse_args([])
        args = apply_yaml_overrides(args, loaded_config)
        
        assert args.dtype == 'bf16'
        print("✅ Quantization algorithm mapping test passed!")
        
    except Exception as e:
        print(f"❌ Quantization mapping test failed: {e}")
    finally:
        os.unlink(yaml_path)

if __name__ == "__main__":
    test_hierarchical_yaml()
    test_quant_algo_mapping()
    print("\n🎉 All hierarchical YAML tests completed successfully!")
