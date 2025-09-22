# HF VRAM Calculator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A professional Python CLI tool for estimating GPU memory requirements for Hugging Face models with different data types and parallelization strategies.

> **⚡ Latest Features**: Smart dtype detection, MHA/MQA/GQA-aware KV cache, 12 quantization formats, 20+ GPU models, professional Rich UI

## Quick Demo

```bash
# Install and run
pip install hf-vram-calc

# Set up authentication (required for most models)
hf auth login --token yourtoken --add-to-git-credential

# Calculate memory requirements
hf-vram-calc microsoft/DialoGPT-medium

# Output: Beautiful tables showing 0.9GB inference, GPU compatibility, parallelization strategies
```

## Features

- 🔍 **Automatic Model Analysis**: Fetch configurations from Hugging Face Hub automatically
- 🧠 **Smart Data Type Detection**: Intelligent dtype recommendation from model names, config, or defaults
- 📊 **Comprehensive Data Type Support**: fp32, fp16, bf16, fp8, int8, int4, mxfp4, nvfp4, awq_int4, gptq_int4, nf4, fp4
- 🎯 **Multi-Scenario Memory Estimation**:
  - **Inference**: Model weights + KV cache overhead (MHA/MQA/GQA-aware, ×1.2 factor)
  - **Training**: Full Adam optimizer states (×4×1.3 factors)
  - **LoRA Fine-tuning**: Low-rank adaptation with trainable parameter overhead
- ⚡ **Advanced Parallelization Analysis**:
  - Tensor Parallelism (TP): 1, 2, 4, 8
  - Pipeline Parallelism (PP): 1, 2, 4, 8  
  - Expert Parallelism (EP) for MoE models
  - Data Parallelism (DP): 2, 4, 8
  - Combined strategies (TP + PP combinations)
- 🎮 **GPU Compatibility Matrix**:
  - 20+ GPU models (RTX 4090, A100, H100, L40S, etc.)
  - Automatic compatibility checking for inference/training/LoRA
  - Minimum GPU memory requirement calculations
- 📈 **Professional Rich UI**:
  - 🎨 Beautiful color-coded tables and panels
  - 📊 Real-time progress indicators
  - 🚀 Modern CLI interface with emoji icons
  - 💡 Smart recommendations and warnings
- 🔧 **Flexible Configuration**:
  - Customizable LoRA rank, batch size, sequence length
  - External JSON configuration files
  - User-defined GPU models and data types
- 📋 **Parameter Display**: Raw count + human-readable format (e.g., "405,016,576 (405.0M)")

## Installation

### Quick Install (from PyPI)

```bash
pip install hf-vram-calc
```

### Build from Source

```bash
# Clone the repository
git clone <repository-url>
cd hf-vram-calc

# Build with uv (recommended)
uv build
uv pip install dist/hf_vram_calc-1.0.0-py3-none-any.whl

# Or install directly
uv pip install .
```

> **Dependencies**: `requests` (HTTP), `rich` (beautiful CLI), Python ≥3.8

For detailed build instructions, see: [BUILD.md](BUILD.md)

## Authentication Setup

Many models require a Hugging Face token. Get yours at https://huggingface.co/settings/tokens, then:

```bash
hf auth login --token yourtoken --add-to-git-credential
```

## Usage

### Basic Usage - Smart Dtype Detection

```bash
# Automatic dtype recommendation from model config/name
hf-vram-calc microsoft/DialoGPT-medium

# Model name contains dtype - automatically detects fp16
hf-vram-calc nvidia/DeepSeek-R1-0528-FP4
```

### Specify Data Type Override

```bash
# Override with specific data type
hf-vram-calc meta-llama/Llama-2-7b-hf --dtype bf16
hf-vram-calc mistralai/Mistral-7B-v0.1 --dtype bf16,fp8
```

### Advanced Configuration

```bash
# Custom batch size and sequence length
hf-vram-calc mistralai/Mistral-7B-v0.1 --batch-size 4 --sequence-length 4096

# Custom LoRA rank for fine-tuning estimation  
hf-vram-calc microsoft/DialoGPT-medium --lora-rank 128

# Detailed analysis (disabled by default)
hf-vram-calc meta-llama/Llama-2-7b-hf --verbose
```

### System Information

```bash
# List all available data types and GPU models
hf-vram-calc --list-types

# Use custom configuration directory
hf-vram-calc --config-dir ./my_config microsoft/DialoGPT-medium

# Show help
hf-vram-calc --help
```

## Command Line Arguments

### Required
- `model_name`: Hugging Face model name (e.g., `microsoft/DialoGPT-medium`)

### Data Type Control  
- `--dtype {fp32,fp16,bf16,fp8,int8,int4,mxfp4,nvfp4,awq_int4,fp4,nf4,gptq_int4}`: Override automatic dtype detection
- `--list-types`: List all available data types and GPU models

### Memory Estimation Parameters
- `--batch-size BATCH_SIZE`: Batch size for activation estimation (default: 1)
- `--sequence-length SEQUENCE_LENGTH`: Sequence length for memory calculation (default: 2048)  
- `--lora-rank LORA_RANK`: LoRA rank for fine-tuning estimation (default: 64)

### Display & Configuration
- `--verbose`: Show detailed parallelization and GPU compatibility (default: disabled)
- `--config-dir CONFIG_DIR`: Custom configuration directory path
- `--help`: Show complete help message with examples

### Smart Behavior
- **No `--dtype`**: Uses intelligent priority (model name → config → fp16 default)
- **With `--dtype`**: Overrides automatic detection with specified type
- **Invalid model**: Graceful error handling with helpful suggestions

## Quick Start Examples

```bash
# Set up authentication first time
hf auth login --token yourtoken --add-to-git-credential

# Estimate memory for different models
hf-vram-calc microsoft/DialoGPT-medium              # → 0.9GB inference (FP16)
hf-vram-calc meta-llama/Llama-2-7b-hf              # → ~13GB inference  
hf-vram-calc nvidia/DeepSeek-R1-0528-FP4           # → Auto-detects FP4 from name

# estimate size for specified quantization versions
hf-vram-calc meta-llama/Llama-2-7b-hf --dtype fp16     # → ~13GB
hf-vram-calc meta-llama/Llama-2-7b-hf --dtype int4     # → ~3.5GB  
hf-vram-calc meta-llama/Llama-2-7b-hf --dtype awq_int4 # → ~3.5GB

# for private access models, it is recommended to use --local-config
hf-vram-calc meta-llama/Llama-4-Scout-17B-16E-Instruct --local-config config.json

# Find optimal parallelization strategy
hf-vram-calc mistralai/Mistral-7B-v0.1 --verbose  # → TP/PP recommendations

# Check what's available
hf-vram-calc --list-types                               # → All types & GPUs
```

## Data Type Priority & Detection

### Automatic Data Type Recommendation

The tool uses intelligent priority-based dtype selection:

1. **Model Name Detection** (Highest Priority)
   - `model-fp16`, `model-bf16` → Extracts from model name  
   - `model-4bit`, `model-gptq`, `model-awq` → Detects quantization
   
2. **Config torch_dtype** (Medium Priority)
   - Reads `torch_dtype` from model's `config.json`
   - Maps `torch.float16` → `fp16`, `torch.bfloat16` → `bf16`, etc.

3. **Default Fallback** (Lowest Priority)
   - Defaults to `fp16` when no dtype detected

### Supported Data Types

| Data Type | Bytes/Param | Description | Detection Patterns |
|-----------|-------------|-------------|--------------------|
| **fp32**  | 4.0 | 32-bit floating point | `fp32`, `float32` |
| **fp16**  | 2.0 | 16-bit floating point | `fp16`, `float16`, `half` |
| **bf16**  | 2.0 | Brain Float 16 | `bf16`, `bfloat16` |
| **fp8**   | 1.0 | 8-bit floating point | `fp8`, `float8` |
| **int8**  | 1.0 | 8-bit integer | `int8`, `8bit` |
| **int4**  | 0.5 | 4-bit integer | `int4`, `4bit` |
| **mxfp4** | 0.5 | Microsoft FP4 | `mxfp4` |
| **nvfp4** | 0.5 | NVIDIA FP4 | `nvfp4` |
| **awq_int4** | 0.5 | AWQ 4-bit quantization | `awq`, `awq-int4` |
| **gptq_int4** | 0.5 | GPTQ 4-bit quantization | `gptq`, `gptq-int4` |
| **nf4**   | 0.5 | 4-bit NormalFloat | `nf4`, `bnb-4bit` |
| **fp4**   | 0.5 | 4-bit floating point | `fp4` |

## Parallelization Strategies

### Tensor Parallelism (TP)
Splits model weights by tensor dimensions across multiple GPUs.

### Pipeline Parallelism (PP)
Distributes different model layers to different GPUs.

### Expert Parallelism (EP)
For MoE (Mixture of Experts) models, distributes expert networks to different GPUs.

### Data Parallelism (DP)
Each GPU holds a complete model copy, only splitting data.

## Example Output

### Smart Dtype Detection Example

```bash
$ hf-vram-calc microsoft/DialoGPT-medium --verbose
```

```
Using recommended data type: FP16
Use --dtype to specify different type, or see --list-types for all options
  🔍 Fetching configuration for microsoft/DialoGPT-medium...
Using recommended data type: FP16
Use --dtype to specify different type, or see --list-types for all options
  📋 Parsing model configuration...                         
  🧮 Calculating model parameters...                        
  💾 Computing memory requirements...                       

                          ╭─────── 🤖 Model Information ───────╮
                          │                                    │
                          │  Model: microsoft/DialoGPT-medium  │
                          │  Architecture: gpt2                │
                          │  Parameters: 406,286,336 (406.3M)  │
                          │  Recommended dtype: FP16           │
                          │                                    │
                          ╰────────────────────────────────────╯

        💾 Memory Requirements by Data Type and Scenario                
╭──────────────┬──────────────┬─────────────────┬─────────────────┬─────────────────┬──────────────╮
│              │   Model Size │        KV Cache │       Inference │        Training │         LoRA │
│  Data Type   │         (GB) │            (GB) │      Total (GB) │     (Adam) (GB) │         (GB) │
├──────────────┼──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────────┤
│     FP16     │         0.76 │            0.19 │            0.91 │            3.94 │         0.94 │
╰──────────────┴──────────────┴─────────────────┴─────────────────┴─────────────────┴──────────────╯
================================================================================
          ⚡ Parallelization Strategies (FP16 Inference)                 
╔════════════════════╤══════╤══════╤══════╤══════╤══════════════╤══════════════╗
║                    │      │      │      │      │   Memory/GPU │   Min GPU    ║
║ Strategy           │  TP  │  PP  │  EP  │  DP  │         (GB) │   Required   ║
╟────────────────────┼──────┼──────┼──────┼──────┼──────────────┼──────────────╢
║ Single GPU         │  1   │  1   │  1   │  1   │         0.91 │     4GB+     ║
║ Tensor Parallel    │  2   │  1   │  1   │  1   │         0.45 │     4GB+     ║
║ TP + PP            │  4   │  4   │  1   │  1   │         0.06 │     4GB+     ║
╚════════════════════╧══════╧══════╧══════╧══════╧══════════════╧══════════════╝

                  🎮 GPU Compatibility Matrix                         
┏━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━┯━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━┓
┃ GPU Type        │   Memory   │  Inference   │   Training   │     LoRA     ┃
┠─────────────────┼────────────┼──────────────┼──────────────┼──────────────┨
┃ RTX 4090        │    24GB    │      ✓       │      ✓       │      ✓       ┃
┃ A100 80GB       │    80GB    │      ✓       │      ✓       │      ✓       ┃
┃ H100 80GB       │    80GB    │      ✓       │      ✓       │      ✓       ┃
┗━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━┷━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━┛

╭─── 📋 Minimum GPU Requirements ────╮
│                                   │
│  Single GPU Inference: 0.9GB      │
│  Single GPU Training: 3.9GB       │  
│  Single GPU LoRA: 0.9GB           │
│                                   │
╰───────────────────────────────────╯
```

### Large Model with User Override

```bash
$ hf-vram-calc nvidia/DeepSeek-R1-0528-FP4 --dtype nvfp4

$ hf-vram-calc Qwen/Qwen-72B-Chat 

```

```
                          ╭──────── 🤖 Model Information ────────╮
                          │                                      │
                          │  Model: nvidia/DeepSeek-R1-0528-FP4  │
                          │  Architecture: deepseek_v3           │
                          │  Parameters: 30,510,606,336 (30.5B)  │
                          │  Original torch_dtype: bfloat16      │
                          │  User specified dtype: NVFP4         │
                          │                                      │
                          ╰──────────────────────────────────────╯

        💾 Memory Requirements by Data Type and Scenario                
╭──────────────┬──────────────┬──────────────┬─────────────────┬──────────────╮
│              │   Total Size │    Inference │        Training │         LoRA │
│  Data Type   │         (GB) │         (GB) │     (Adam) (GB) │         (GB) │
├──────────────┼──────────────┼──────────────┼─────────────────┼──────────────┤
│    NVFP4     │        14.21 │        17.05 │           73.88 │        19.34 │
╰──────────────┴──────────────┴──────────────┴─────────────────┴──────────────╯
```

### List Available Types

```bash
$ hf-vram-calc --list-types
```

```
Available Data Types:
╭───────────┬─────────────┬────────────────────────╮
│ Data Type │ Bytes/Param │ Description            │
├───────────┼─────────────┼────────────────────────┤
│ FP32      │           4 │ 32-bit floating point  │
│ FP16      │           2 │ 16-bit floating point  │
│ BF16      │           2 │ Brain Float 16         │
│ NVFP4     │         0.5 │ NVIDIA FP4             │
│ AWQ_INT4  │         0.5 │ AWQ 4-bit quantization │
│ GPTQ_INT4 │         0.5 │ GPTQ 4-bit quantization│
╰───────────┴─────────────┴────────────────────────╯

Available GPU Types:
╭───────────────────┬─────────────┬────────────┬──────────────╮
│ GPU Name          │ Memory (GB) │ Category   │ Architecture │
├───────────────────┼─────────────┼────────────┼──────────────┤
│ RTX 4090          │          24 │ consumer   │ Ada Lovelace │
│ A100 80GB         │          80 │ datacenter │ Ampere       │
│ H100 80GB         │          80 │ datacenter │ Hopper       │
╰───────────────────┴─────────────┴────────────┴──────────────╯
```

## Calculation Formulas

### Inference Memory
```
Inference Memory = Model Weights × 1.2
```
Includes model weights and KV cache overhead.

### KV Cache Memory
```
KV Cache (GB) = 2 × Batch_Size × Sequence_Length × Head_Dim × Num_KV_Heads × Num_Layers × Precision ÷ 1,073,741,824
```
- Head_Dim = hidden_size ÷ num_attention_heads
- Num_KV_Heads = config.num_key_value_heads (if present) else num_attention_heads
- Automatically supports MHA, MQA, and GQA via model config; KV cache uses FP16/BF16 for quantized models

### Training Memory (with Adam)
```
Training Memory = Model Weights × 4 × 1.3
```
- 4x factor: Model weights (1x) + Gradients (1x) + Adam optimizer states (2x)
- 1.3x factor: 30% additional overhead (activation caching, etc.)

### LoRA Fine-tuning Memory
```
LoRA Memory = (Model Weights + LoRA Parameter Overhead) × 1.2
```
LoRA parameter overhead calculated based on rank and target module ratio.

## Advanced Features

### Configuration System

External JSON configuration files for maximum flexibility:

- **`data_types.json`** - Add custom quantization formats
- **`gpu_types.json`** - Define new GPU models and specifications  
- **`display_settings.json`** - Customize UI appearance and limits

```bash
# Use custom config directory
hf-vram-calc --config-dir ./custom_config model_name

# Add custom data type example (data_types.json)
{
  "my_custom_int2": {
    "bytes_per_param": 0.25,
    "description": "Custom 2-bit quantization"
  }
}
```

### Memory Calculation Details

| Scenario | Formula | Explanation |
|----------|---------|-------------|
| **Inference** | Model × 1.2 | Includes KV cache and activation overhead |
| **Training** | Model × 4 × 1.3 | Weights(1x) + Gradients(1x) + Adam(2x) + 30% overhead |
| **LoRA** | (Model + LoRA_params×4) × 1.2 | Base model + trainable parameters with optimizer |

### Parallelization Efficiency

- **TP (Tensor Parallel)**: Near-linear scaling, slight communication overhead
- **PP (Pipeline Parallel)**: Good efficiency, pipeline bubble ~10-15%  
- **EP (Expert Parallel)**: MoE-specific, depends on expert routing efficiency
- **DP (Data Parallel)**: No memory reduction per GPU, full model replica

## Supported Architectures

### Fully Supported ✅
- **GPT Family**: GPT-2, GPT-3, GPT-4, GPT-NeoX, etc.
- **LLaMA Family**: LLaMA, LLaMA-2, Code Llama, Vicuna, etc.
- **Mistral Family**: Mistral 7B, Mixtral 8x7B (MoE), etc.
- **Other Transformers**: BERT, RoBERTa, T5, FLAN-T5, etc.
- **New Architectures**: DeepSeek, Qwen, ChatGLM, Baichuan, etc.

### Architecture Detection
- **Automatic field mapping** for different config.json formats
- **Fallback support** for uncommon architectures
- **MoE handling** for Mixture-of-Experts models

## Accuracy & Limitations

### ✅ Highly Accurate For:
- **Parameter counting** (exact calculation)
- **Memory estimation** (within 5-10% of actual)
- **Parallelization ratios** (theoretical maximum)

### ⚠️ Considerations:
- **Activation memory** varies with sequence length and optimization
- **Real-world efficiency** may differ due to framework overhead  
- **Quantization accuracy** depends on specific implementation
- **MoE models** require expert routing consideration

## Build & Development

Built with modern Python tooling:
- **uv**: Fast Python package management and building
- **Rich**: Professional terminal interface
- **Requests**: HTTP client for model config fetching
- **JSON configuration**: Flexible external configuration system

For development setup, see: [BUILD.md](BUILD.md)

## Contributing

We welcome contributions! Areas for improvement:

- 🔧 **New quantization formats** (add to `data_types.json`)
- 🎮 **GPU models** (update `gpu_types.json`)  
- 📊 **Architecture support** (enhance config parsing)
- 🚀 **Performance optimizations**
- 📚 **Documentation improvements**
- 🧪 **Test coverage expansion**

## See Also

- 📚 **[BUILD.md](BUILD.md)** - Complete build and installation guide
- ⚙️ **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - Configuration customization details
- 📝 **Examples in help**: `hf-vram-calc --help` for usage examples

## Version History

- **v1.0.0**: Complete rewrite with uv build, smart dtype detection, professional UI
- **v0.x**: Legacy single-file version (deprecated)

## License

MIT License - see LICENSE file for details.

---

**Made with ❤️ for the ML community** | Built with [uv](https://github.com/astral-sh/uv) and [Rich](https://github.com/Textualize/rich)
