# Configuration Guide

VRAM Calculator uses a separate JSON configuration file system that allows users to flexibly add new data types and GPU models.

## Configuration File Structure

The system uses three independent JSON configuration files:

### 1. `data_types.json` - Data Type Configuration

Used to define bytes per parameter for different data types:

```json
{
  "fp32": {
    "bytes_per_param": 4,
    "description": "32-bit floating point"
  },
  "your_custom_type": {
    "bytes_per_param": 1.5,
    "description": "Your custom quantization format"
  }
}
```

**Field Descriptions:**
- `bytes_per_param`: Bytes per parameter (supports decimals)
- `description`: Data type description

### 2. `gpu_types.json` - GPU Type Configuration

Used to define GPU models and specifications:

```json
[
  {
    "name": "RTX 4090",
    "memory_gb": 24,
    "category": "consumer",
    "architecture": "Ada Lovelace"
  },
  {
    "name": "Your Custom GPU",
    "memory_gb": 32,
    "category": "custom",
    "architecture": "Custom Arch"
  }
]
```

**Field Descriptions:**
- `name`: GPU model name
- `memory_gb`: GPU memory size (GB)
- `category`: GPU category (consumer/datacenter/custom)
- `architecture`: GPU architecture (optional)

### 3. `display_settings.json` - Display Configuration

Used to control interface display behavior:

```json
{
  "max_gpu_display": 8,
  "preferred_categories": ["datacenter", "consumer"],
  "memory_color_thresholds": {
    "green": 8,
    "yellow": 24,
    "red": 80
  },
  "table_styles": {
    "memory_table": {
      "box": "ROUNDED",
      "show_lines": true
    }
  }
}
```

## Usage

### View Current Configuration

```bash
python3 vram_calculator.py --list-types
```

### Add New Data Type

1. Edit the `data_types.json` file
2. Add new data type entry:

```json
{
  "my_new_format": {
    "bytes_per_param": 0.75,
    "description": "My custom 6-bit format"
  }
}
```

3. Use the new data type:

```bash
python3 vram_calculator.py --dtype my_new_format microsoft/DialoGPT-medium
```

### Add New GPU Model

1. Edit the `gpu_types.json` file
2. Add new GPU entry to the array:

```json
{
  "name": "RTX 5090",
  "memory_gb": 32,
  "category": "consumer",
  "architecture": "Blackwell"
}
```

### Use Custom Configuration Directory

```bash
# Place configuration files in custom directory
mkdir my_config
cp *.json my_config/

# Use custom configuration directory
python3 vram_calculator.py --config-dir my_config microsoft/DialoGPT-medium
```

## Common Data Type Examples

```json
{
  "fp32": {"bytes_per_param": 4, "description": "32-bit floating point"},
  "fp16": {"bytes_per_param": 2, "description": "16-bit floating point"},
  "bf16": {"bytes_per_param": 2, "description": "Brain Float 16"},
  "int8": {"bytes_per_param": 1, "description": "8-bit integer"},
  "int4": {"bytes_per_param": 0.5, "description": "4-bit integer"},
  "awq_int4": {"bytes_per_param": 0.5, "description": "AWQ 4-bit quantization"},
  "gptq_int4": {"bytes_per_param": 0.5, "description": "GPTQ 4-bit quantization"},
  "qlora_int4": {"bytes_per_param": 0.5, "description": "QLoRA 4-bit quantization"},
  "int2": {"bytes_per_param": 0.25, "description": "2-bit integer"},
  "int1": {"bytes_per_param": 0.125, "description": "1-bit integer"}
}
```

## Configuration File Validation

The system automatically validates configuration file formats:

- `data_types.json`: Must contain `bytes_per_param` field
- `gpu_types.json`: Must contain `name` and `memory_gb` fields
- If configuration files don't exist, the system will create default configurations

## Error Handling

If configuration files have problems, the system will display detailed error messages:

```bash
❌ Invalid JSON in data_types.json: Expecting ',' delimiter: line 5 column 3 (char 123)
```

## Backup Recommendations

It's recommended to regularly backup your configuration files:

```bash
# Create backups
cp data_types.json data_types.json.backup
cp gpu_types.json gpu_types.json.backup
cp display_settings.json display_settings.json.backup
```

## Advanced Configuration

### Custom Color Thresholds

Modify `display_settings.json` to change memory usage color coding:

```json
{
  "memory_color_thresholds": {
    "green": 4,    // <= 4GB shows green
    "yellow": 16,  // <= 16GB shows yellow
    "red": 64      // > 64GB shows red
  }
}
```

### Table Style Customization

Customize table appearance:

```json
{
  "table_styles": {
    "memory_table": {
      "box": "HEAVY_EDGE",
      "show_lines": false,
      "header_style": "bold red"
    }
  }
}
```

Available box styles: `ROUNDED`, `DOUBLE_EDGE`, `HEAVY_EDGE`, `SIMPLE`, `ASCII`

### GPU Display Filtering

Control which GPUs are displayed:

```json
{
  "max_gpu_display": 6,
  "preferred_categories": ["datacenter"],  // Only show datacenter GPUs
  "memory_color_thresholds": {
    "green": 16,
    "yellow": 40,
    "red": 80
  }
}
```

## Configuration Examples for Different Use Cases

### For Research/Academic Use

Focus on common research GPUs:

```json
{
  "max_gpu_display": 5,
  "preferred_categories": ["datacenter"],
  "gpu_types": [
    {"name": "V100 32GB", "memory_gb": 32, "category": "datacenter"},
    {"name": "A100 40GB", "memory_gb": 40, "category": "datacenter"},
    {"name": "A100 80GB", "memory_gb": 80, "category": "datacenter"},
    {"name": "H100 80GB", "memory_gb": 80, "category": "datacenter"}
  ]
}
```

### For Consumer Use

Focus on gaming/prosumer GPUs:

```json
{
  "max_gpu_display": 8,
  "preferred_categories": ["consumer"],
  "memory_color_thresholds": {
    "green": 8,
    "yellow": 16,
    "red": 24
  }
}
```

### For Custom Quantization Research

Add experimental quantization formats:

```json
{
  "experimental_int3": {
    "bytes_per_param": 0.375,
    "description": "Experimental 3-bit integer"
  },
  "custom_fp5": {
    "bytes_per_param": 0.625,
    "description": "Custom 5-bit floating point"
  },
  "binary_int1": {
    "bytes_per_param": 0.125,
    "description": "Binary weights (1-bit)"
  }
}
```