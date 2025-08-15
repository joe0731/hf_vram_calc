# Build Guide

This guide explains how to build and install the HF VRAM Calculator CLI tool using `uv`.

## Prerequisites

Make sure you have `uv` installed. If not, install it:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Or using brew (macOS)
brew install uv
```

## Building from Source

### 1. Clone the Repository

```bash
git clone <repository-url>
cd hf-vram-calc
```

### 2. Build the Package

Using `uv` to build the package:

```bash
# Build wheel and source distribution
uv build

# Or build only wheel
uv build --wheel

# Or build only source distribution  
uv build --sdist
```

The built packages will be available in the `dist/` directory.

### 3. Install the Package

#### Option A: Install from Built Wheel

```bash
# Install the built wheel
uv pip install dist/hf_vram_calc-1.0.0-py3-none-any.whl

# Or install in editable mode for development
uv pip install -e .
```

#### Option B: Install Directly from Source

```bash
# Install directly from source
uv pip install .

# Or for development with dependencies
uv pip install -e ".[dev]"
```

#### Option C: Create Virtual Environment and Install

```bash
# Create a new virtual environment
uv venv hf-vram-calc-env

# Activate the environment
source hf-vram-calc-env/bin/activate  # Linux/macOS
# Or: hf-vram-calc-env\Scripts\activate  # Windows

# Install the package
uv pip install .
```

## Usage

After installation, the `hf-vram-calc` command will be available:

```bash
# Show help
hf-vram-calc --help

# List available data types and GPU models
hf-vram-calc --list-types

# Calculate memory for a model
hf-vram-calc microsoft/DialoGPT-medium

# Calculate for specific data type
hf-vram-calc --dtype bf16 microsoft/DialoGPT-medium

# Show detailed analysis
hf-vram-calc --show-detailed microsoft/DialoGPT-medium
```

## Development

### Setting up Development Environment

```bash
# Create virtual environment for development
uv venv

# Activate environment
source .venv/bin/activate

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Install test dependencies
uv pip install pytest

# Run tests (when available)
pytest
```

### Code Formatting

```bash
# Format code with black
black src/

# Sort imports with isort
isort src/

# Check code style with flake8
flake8 src/
```

## Configuration

The tool uses configuration files in the package directory:

- `data_types.json` - Data type definitions
- `gpu_types.json` - GPU model specifications
- `display_settings.json` - Display preferences

You can override these by:

1. Creating your own configuration directory
2. Using `--config-dir /path/to/your/config` option

### Custom Configuration Example

```bash
# Create custom config directory
mkdir my-config
cp src/hf_vram_calc/*.json my-config/

# Edit configuration files
vim my-config/data_types.json

# Use custom configuration
hf-vram-calc --config-dir my-config microsoft/DialoGPT-medium
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're in the correct environment:

```bash
# Check if hf-vram-calc is installed
which hf-vram-calc

# Check Python path
python -c "import hf_vram_calc; print(hf_vram_calc.__file__)"
```

### Permission Issues

If you get permission errors during installation:

```bash
# Install for current user only
uv pip install --user .

# Or use virtual environment (recommended)
uv venv
source .venv/bin/activate
uv pip install .
```

### Configuration File Errors

If configuration files are not found:

```bash
# Check package installation location
python -c "import hf_vram_calc; print(hf_vram_calc.__file__)"

# Copy configuration files manually if needed
cp *.json /path/to/installed/package/
```

## Building for Distribution

### Create Source Distribution

```bash
uv build --sdist
```

### Create Wheel Distribution

```bash
uv build --wheel
```

### Build Both

```bash
uv build
```

### Upload to PyPI (when ready)

```bash
# Install twine
uv pip install twine

# Upload to test PyPI first
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Environment Variables

You can set these environment variables for customization:

- `HF_VRAM_CALC_CONFIG_DIR` - Default configuration directory
- `HF_VRAM_CALC_CACHE_DIR` - Cache directory for model configs

## Performance Tips

1. **Use virtual environments** to avoid dependency conflicts
2. **Install in editable mode** (`-e`) during development
3. **Use `--config-dir`** for custom configurations instead of modifying package files
4. **Cache model configurations** to avoid repeated downloads

## Support

For issues and questions:

1. Check this BUILD.md file
2. Review the main README.md
3. Check CONFIG_GUIDE.md for configuration help
4. Open an issue on the repository

## Advanced Usage

### Custom Data Types

Add new quantization formats by editing `data_types.json`:

```json
{
  "my_custom_format": {
    "bytes_per_param": 0.75,
    "description": "My custom 6-bit format"
  }
}
```

### Custom GPU Models

Add new GPU models by editing `gpu_types.json`:

```json
{
  "name": "RTX 5090",
  "memory_gb": 32,
  "category": "consumer",
  "architecture": "Blackwell"
}
```

### Scripting Integration

Use the tool in scripts:

```bash
#!/bin/bash
MODEL="microsoft/DialoGPT-medium"
DTYPE="bf16"

# Get memory requirements
OUTPUT=$(hf-vram-calc --dtype $DTYPE $MODEL)
echo "Memory requirements for $MODEL with $DTYPE:"
echo "$OUTPUT"
```
