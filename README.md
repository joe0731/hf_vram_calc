# Model VRAM Calculator

A Python CLI tool for estimating GPU memory requirements for Hugging Face models with different data types and parallelization strategies.

## Features

- 🔍 Automatically fetch model configurations from Hugging Face
- 📊 Support multiple data types: fp32, fp16/bf16, fp8, int8, int4, mxfp4, nvfp4
- 🎯 Memory estimation for different scenarios:
  - **Inference**: Model weights + KV cache overhead
  - **Training**: Including gradients and optimizer states (Adam)
  - **LoRA Fine-tuning**: Low-rank adaptation fine-tuning memory requirements
- ⚡ Calculate memory distribution across parallelization strategies:
  - Tensor Parallelism (TP): 1, 2, 4, 8
  - Pipeline Parallelism (PP): 1, 2, 4, 8
  - Expert Parallelism (EP)
  - Data Parallelism (DP)
  - Combined strategies (TP + PP)
- 🎮 GPU compatibility checks:
  - Common GPU type recommendations (RTX 4090, A100, H100, etc.)
  - Minimum GPU memory requirement calculations
- 📈 Professional table output using Rich library:
  - 🎨 Color coding and beautiful borders
  - 📊 Progress bars and status displays
  - 🚀 Modern CLI interface experience
- 🔧 Customizable parameters: LoRA rank, batch size, sequence length

## Installation

```bash
pip3 install -r requirements.txt
```

> Main dependencies: `requests` and `rich` (for beautiful tables and progress display)

## Usage

### Basic Usage

```bash
python3 vram_calculator.py microsoft/DialoGPT-medium
```

### Specify Data Type

```bash
python3 vram_calculator.py meta-llama/Llama-2-7b-hf --dtype bf16
```

### Custom Batch Size and Sequence Length

```bash
python3 vram_calculator.py mistralai/Mistral-7B-v0.1 --batch-size 4 --sequence-length 4096
```

### Show Detailed Parallelization Strategies and GPU Recommendations

```bash
python3 vram_calculator.py --show-detailed microsoft/DialoGPT-medium
```

### Custom LoRA Rank for Fine-tuning Memory Estimation

```bash
python3 vram_calculator.py --lora-rank 128 --show-detailed microsoft/DialoGPT-medium
```

### View Available Data Types and GPU Models

```bash
python3 vram_calculator.py --list-types
```

### Use Custom Configuration

```bash
# Use custom configuration directory
python3 vram_calculator.py --config-dir ./my_config microsoft/DialoGPT-medium
```

## Command Line Arguments

- `model_name`: Hugging Face model name (required)
- `--dtype`: Specify data type (optional, default: show all types)
- `--batch-size`: Batch size for activation memory estimation (default: 1)
- `--sequence-length`: Sequence length for activation memory estimation (default: 2048)
- `--lora-rank`: LoRA rank parameter for fine-tuning (default: 64)
- `--show-detailed`: Show detailed parallelization strategies and GPU recommendations
- `--config-dir`: Specify custom configuration directory
- `--list-types`: List all available data types and GPU models

## Configuration System

The tool uses separate JSON configuration files to manage data types and GPU specifications, allowing flexible user customization:

### Configuration File Structure

- **`data_types.json`** - Define data types and bytes per parameter
- **`gpu_types.json`** - Define GPU models and memory specifications  
- **`display_settings.json`** - Control display styles and behavior

### Adding Custom Data Types

Edit the `data_types.json` file:

```json
{
  "your_custom_format": {
    "bytes_per_param": 0.75,
    "description": "Your custom 6-bit format"
  }
}
```

### Adding Custom GPU Models

Edit the `gpu_types.json` file:

```json
{
  "name": "RTX 5090",
  "memory_gb": 32,
  "category": "consumer",
  "architecture": "Blackwell"
}
```

For detailed configuration instructions, please refer to: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

## Supported Data Types

| Data Type | Bytes per Parameter | Description |
|-----------|--------------------|-----------| 
| fp32      | 4                  | 32-bit floating point |
| fp16      | 2                  | 16-bit floating point |
| bf16      | 2                  | Brain Float 16 |
| fp8       | 1                  | 8-bit floating point |
| int8      | 1                  | 8-bit integer |
| int4      | 0.5                | 4-bit integer |
| mxfp4     | 0.5                | Microsoft FP4 |
| nvfp4     | 0.5                | NVIDIA FP4 |

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

### Basic Output (Default Mode)

```
================================================================================
Model: microsoft/DialoGPT-medium
Architecture: gpt2
Parameters: 404,966,400
================================================================================

Memory Requirements by Data Type and Scenario:              
================================================================================
Data Type    Total Size   Inference    Training     LoRA        
(GB)         (GB)         (GB)         (Adam) (GB)  (GB)        
────────────────────────────────────────────────────────────────────────────────
FP32         1.51        1.81        7.84        1.84       
FP16         0.75        0.91        3.92        0.94       
BF16         0.75        0.91        3.92        0.94       
INT8         0.38        0.45        1.96        0.48       
INT4         0.19        0.23        0.98        0.26       
```

### Detailed Output (--show-detailed mode)

```
================================================================================
Model: microsoft/DialoGPT-medium
Architecture: gpt2
Parameters: 404,966,400
================================================================================

Memory Requirements by Data Type and Scenario:              
================================================================================
Data Type    Total Size   Inference    Training     LoRA        
(GB)         (GB)         (GB)         (Adam) (GB)  (GB)        
────────────────────────────────────────────────────────────────────────────────
FP32         1.51        1.81        7.84        1.84       
FP16         0.75        0.91        3.92        0.94       
BF16         0.75        0.91        3.92        0.94       
INT8         0.38        0.45        1.96        0.48       
INT4         0.19        0.23        0.98        0.26       

Parallelization Strategies (BF16 Inference):                
================================================================================
Strategy             TP   PP   EP   DP   Memory/GPU (GB) Min GPUs  
────────────────────────────────────────────────────────────────────────────────
Single GPU           1    1    1    1    0.91           4GB+      
Tensor Parallel      2    1    1    1    0.45           4GB+      
Tensor Parallel      4    1    1    1    0.23           4GB+      
Tensor Parallel      8    1    1    1    0.11           4GB+      
Pipeline Parallel    1    2    1    1    0.45           4GB+      
Pipeline Parallel    1    4    1    1    0.23           4GB+      
Pipeline Parallel    1    8    1    1    0.11           4GB+      
TP + PP              2    2    1    1    0.23           4GB+      
TP + PP              2    4    1    1    0.11           4GB+      
TP + PP              4    2    1    1    0.11           4GB+      
TP + PP              4    4    1    1    0.06           4GB+      

Recommendations:                                            
================================================================================
GPU Type        Memory     Inference    Training     LoRA        
────────────────────────────────────────────────────────────────────────────────
RTX 4090        24       GB ✓           ✓           ✓          
A100 40GB       40       GB ✓           ✓           ✓          
A100 80GB       80       GB ✓           ✓           ✓          
H100            80       GB ✓           ✓           ✓          

Minimum GPU Requirements:                                   
────────────────────────────────────────────────────────────────────────────────
Single GPU Inference: 0.9GB
Single GPU Training: 3.9GB
Single GPU LoRA: 0.9GB
```

## Calculation Formulas

### Inference Memory
```
Inference Memory = Model Weights × 1.2
```
Includes model weights and KV cache overhead.

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

## Notes

1. **Activation Memory**: Current simplified estimation may be significantly reduced in practice due to optimization strategies (such as gradient checkpointing)
2. **Parallelization Efficiency**: Assumes ideal conditions, actual may vary slightly due to communication overhead
3. **LoRA Estimation**: Based on typical configurations (25% target modules), actual may vary depending on specific implementation
4. **Mixed Data Types**: Some cases may use mixed precision, actual memory between different data types
5. **Model Architecture Differences**: Different architectures (such as MoE) may have special memory distribution patterns

## Supported Model Architectures

Currently mainly supports Transformer architecture models, including but not limited to:
- GPT series
- LLaMA series
- Mistral series
- BERT series
- T5 series

## Contributing

Welcome to submit Issues and Pull Requests to improve this tool!