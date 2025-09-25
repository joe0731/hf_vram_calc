"""
Command-line interface for HF VRAM Calculator.
"""

import argparse
import sys
from typing import Dict

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.align import Align
from rich import box

from .config import ConfigManager
from .parser import ConfigParser
from .calculator import VRAMCalculator, ParameterCalculator, LlmodelMemoryResult
from .parallel import ParallelizationCalculator
from .models import ModelConfig
from .__init__ import __version__

# Create global console instance
console = Console()


def format_memory_size(memory_gb: float) -> str:
    """Format memory size with appropriate unit"""
    if memory_gb >= 1024:
        return f"{memory_gb / 1024:.2f} TB"
    else:
        return f"{memory_gb:.2f} GB"


def format_parameters(num_params: int) -> str:
    """Format parameter count with B (Billions) unit"""
    if num_params >= 1_000_000_000:
        billions = num_params / 1_000_000_000
        if billions >= 100:
            return f"{billions:.0f}B"  # 100B+ no decimal
        elif billions >= 10:
            return f"{billions:.1f}B"  # 10.5B one decimal
        else:
            return f"{billions:.2f}B"  # 2.34B two decimals
    elif num_params >= 1_000_000:
        millions = num_params / 1_000_000
        return f"{millions:.1f}M"
    elif num_params >= 1_000:
        thousands = num_params / 1_000
        return f"{thousands:.1f}K"
    else:
        return str(num_params)


def print_memory_table(memory_results: list[LlmodelMemoryResult], batch_size: int = 1, sequence_length: int = 2048):
    """Print memory requirements table by data type and scenario"""
    console.print()
    
    # Create beautiful table
    table = Table(
        title=f"💾 Memory Requirements by Data Type and Scenario (Batch Size: {batch_size}, Sequence Length: {sequence_length})",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold blue",
        show_lines=True
    )
    
    # Add columns
    table.add_column("Data Type", justify="center", style="cyan", width=12)
    table.add_column("Model Size\n(GB)", justify="right", style="green", width=12)
    table.add_column("KV Cache\n(GB)", justify="right", style="magenta", width=15)
    table.add_column("Inference\nTotal (GB)", justify="right", style="yellow", width=15)
    table.add_column("Training\n(Adam) (GB)", justify="right", style="red", width=15)
    table.add_column("LoRA\n(GB)", justify="right", style="blue", width=12)
    
    # Add rows - display all memory results
    for memory_result in sorted(memory_results, key=lambda x: x.dtype):
        # Format numbers with appropriate colors
        model_str = f"{memory_result.base_memory:.2f}"
        kv_cache_str = f"{memory_result.kv_cache_memory:.2f}"
        inference_str = f"{memory_result.inference_memory:.2f}"
        training_str = f"{memory_result.training_memory:.2f}"
        lora_str = f"{memory_result.lora_memory:.2f}"
        
        table.add_row(
            memory_result.dtype.upper(),
            model_str,
            kv_cache_str,
            inference_str,
            training_str,
            lora_str
        )
    
    console.print(table)


def print_parallelization_table(memory_results: list[LlmodelMemoryResult]):
    """Print parallelization strategies table"""
    console.print()
    
    if not memory_results:
        return
    
    # Prefer bf16 or fp16, otherwise use the first available
    preferred_result = None
    for result in memory_results:
        if result.dtype in ['bf16', 'fp16', 'fp32']:
            preferred_result = result
            break
    
    if preferred_result is None:
        preferred_result = memory_results[0]
    
    # Use the total inference memory (model + overhead, not including KV cache for parallelization)
    inference_memory = preferred_result.inference_memory
    
    # Create beautiful parallelization table
    table = Table(
        title=f"⚡ Parallelization Strategies ({preferred_result.dtype.upper()} Inference)",
        box=box.DOUBLE_EDGE,
        header_style="bold cyan",
        title_style="bold green",
        show_lines=True
    )
    
    # Add columns
    table.add_column("Strategy", justify="left", style="bright_white", width=18)
    table.add_column("TP", justify="center", style="cyan", width=4)
    table.add_column("PP", justify="center", style="magenta", width=4)
    table.add_column("EP", justify="center", style="yellow", width=4)
    table.add_column("DP", justify="center", style="blue", width=4)
    table.add_column("Memory/GPU\n(GB)", justify="right", style="green", width=12)
    table.add_column("Min GPU\nRequired", justify="center", style="red", width=12)
    
    # Common GPU memory sizes for reference
    gpu_sizes = [4, 8, 16, 24, 40, 80]  # common GPU memory sizes in GB
    
    strategies = [
        ("Single GPU", 1, 1, 1, 1),
        ("Tensor Parallel", 2, 1, 1, 1),
        ("Tensor Parallel", 4, 1, 1, 1),
        ("Tensor Parallel", 8, 1, 1, 1),
        ("Pipeline Parallel", 1, 2, 1, 1),
        ("Pipeline Parallel", 1, 4, 1, 1),
        ("Pipeline Parallel", 1, 8, 1, 1),
        ("TP + PP", 2, 2, 1, 1),
        ("TP + PP", 2, 4, 1, 1),
        ("TP + PP", 4, 2, 1, 1),
        ("TP + PP", 4, 4, 1, 1),
        ("Data Parallel", 1, 1, 1, 2),
        ("Data Parallel", 1, 1, 1, 4),
        ("Data Parallel", 1, 1, 1, 8),
    ]
    
    for strategy_name, tp, pp, ep, dp in strategies:
        memory_per_gpu = ParallelizationCalculator.calculate_combined_parallel(
            inference_memory, tp, pp, ep, dp
        )
        
        # Find minimum GPU memory requirement and add color coding
        suitable_gpu = None
        gpu_style = "red"
        for gpu_size in gpu_sizes:
            if memory_per_gpu <= gpu_size:
                suitable_gpu = f"{gpu_size}GB+"
                if gpu_size <= 8:
                    gpu_style = "green"
                elif gpu_size <= 24:
                    gpu_style = "yellow"
                else:
                    gpu_style = "red"
                break
        
        if suitable_gpu is None:
            suitable_gpu = ">80GB"
            gpu_style = "bright_red"
        
        # Color code memory usage
        memory_style = "green" if memory_per_gpu < 8 else "yellow" if memory_per_gpu < 24 else "red"
        
        table.add_row(
            strategy_name,
            str(tp),
            str(pp), 
            str(ep),
            str(dp),
            f"[{memory_style}]{memory_per_gpu:.2f}[/{memory_style}]",
            f"[{gpu_style}]{suitable_gpu}[/{gpu_style}]"
        )
    
    console.print(table)


def print_detailed_recommendations(memory_results: list[LlmodelMemoryResult]):
    """Print detailed recommendations"""
    if not memory_results:
        return
    
    # Prefer bf16 or fp16, otherwise use the first available
    preferred_result = None
    for result in memory_results:
        if result.dtype in ['bf16', 'fp16', 'fp32']:
            preferred_result = result
            break
    
    if preferred_result is None:
        preferred_result = memory_results[0]
    
    # Use stored values from MemoryResult
    inference_memory = preferred_result.inference_memory
    training_memory = preferred_result.training_memory
    lora_memory = preferred_result.lora_memory
    
    console.print()
    
    # GPU compatibility table
    gpu_table = Table(
        title="🎮 GPU Compatibility Matrix",
        box=box.HEAVY_EDGE,
        header_style="bold white",
        title_style="bold cyan",
        show_lines=True
    )
    
    gpu_table.add_column("GPU Type", justify="left", style="bright_white", width=15)
    gpu_table.add_column("Memory", justify="center", style="cyan", width=10)
    gpu_table.add_column("Inference", justify="center", style="green", width=12)
    gpu_table.add_column("Training", justify="center", style="red", width=12)
    gpu_table.add_column("LoRA", justify="center", style="blue", width=12)
    
    # GPU recommendations from configuration
    config_manager = ConfigManager()
    gpu_recommendations = config_manager.get_gpu_types()
    display_settings = config_manager.get_display_settings()
    # remove max display limit to show all GPUs
    max_display = len(gpu_recommendations)
    
    # Limit number of GPUs displayed
    displayed_gpus = gpu_recommendations[:max_display]
    
    for gpu_name, gpu_memory, category in displayed_gpus:
        # Check what's possible with single GPU and color code
        can_inference = "[green]✓[/green]" if inference_memory <= gpu_memory else "[red]✗[/red]"
        can_training = "[green]✓[/green]" if training_memory <= gpu_memory else "[red]✗[/red]"
        can_lora = "[green]✓[/green]" if lora_memory <= gpu_memory else "[red]✗[/red]"
        
        # Color code GPU memory based on availability
        memory_style = "green" if gpu_memory >= 40 else "yellow" if gpu_memory >= 16 else "red"
        
        gpu_table.add_row(
            gpu_name,
            f"[{memory_style}]{gpu_memory}GB[/{memory_style}]",
            can_inference,
            can_training,
            can_lora
        )
    
    console.print(gpu_table)
    
    # Minimum requirements panel
    console.print()
    requirements_text = f"""
[bold green]Single GPU Inference:[/bold green] {inference_memory:.1f}GB
[bold red]Single GPU Training:[/bold red] {training_memory:.1f}GB  
[bold blue]Single GPU LoRA:[/bold blue] {lora_memory:.1f}GB
    """
    
    requirements_panel = Panel(
        requirements_text.strip(),
        title="📋 Minimum GPU Requirements",
        border_style="bright_blue",
        padding=(1, 2)
    )
    
    console.print(requirements_panel)


def print_model_header(config: ModelConfig, num_params: int, user_specified_dtype: str = None):
    """Print beautiful model information header"""
    console.print()
    
    # Format parameters in both raw and human-readable format
    params_formatted = format_parameters(num_params)
    
    # Create model info panel with dtype info
    model_info = f"""
[bold]Model:[/bold] [cyan]{config.model_name}[/cyan]
[bold]Architecture:[/bold] [magenta]{config.model_type}[/magenta]
[bold]Parameters:[/bold] [green]{num_params:,}[/green] [dim]({params_formatted})[/dim]"""
    
    # Add torch_dtype info if available
    if config.torch_dtype:
        model_info += f"\n[bold]Original torch_dtype:[/bold] [yellow]{config.torch_dtype}[/yellow]"
    
    # Show user specified dtype or recommended dtype
    if user_specified_dtype:
        model_info += f"\n[bold]User specified dtype:[/bold] [bright_blue]{user_specified_dtype.upper()}[/bright_blue]"
    elif config.recommended_dtype:
        model_info += f"\n[bold]Recommended dtype:[/bold] [bright_green]{config.recommended_dtype.upper()}[/bright_green]"
    
    header_panel = Panel(
        model_info.strip(),
        title="🤖 Model Information",
        border_style="bright_cyan",
        padding=(1, 2),
        expand=False
    )
    
    console.print(Align.center(header_panel))


def print_results(config: ModelConfig, memory_results: list[LlmodelMemoryResult], verbose: bool = True):
    """Print comprehensive formatted results"""
    if not memory_results:
        return
    # Use the first result for num_params (all should have the same value)
    num_params = memory_results[0].num_params
    # Print model header
    all_dtypes = ",".join([result.dtype.upper() for result in memory_results])
    print_model_header(config, num_params, all_dtypes)
    # Print main memory table
    print_memory_table(memory_results, memory_results[0].batch_size, memory_results[0].sequence_length)

    # Only print detailed info if verbose is True
    if verbose:
        # Print detailed info for each dtype
        for memory_result in memory_results:
            # Add separator line between dtypes (except for the first one)
            console.print("\n" + "="*80 + "\n")
            # Print parallelization table for this dtype
            print_parallelization_table([memory_result])

            # Print detailed recommendations for this dtype
            print_detailed_recommendations([memory_result])


def main():
    """Main CLI entry point"""
    # Initialize config manager early to get available types
    temp_config_manager = ConfigManager()
    available_dtypes = list(temp_config_manager.get_data_types().keys())
    
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory requirements for Hugging Face models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hf-vram-calc microsoft/DialoGPT-medium
  hf-vram-calc meta-llama/Llama-2-7b-hf
  hf-vram-calc mistralai/Mistral-7B-v0.1
  hf-vram-calc --list-types  # show available data types and GPUs
  hf-vram-calc my-model --local-config /path/to/config.json  # use local config file
        """
    )
    
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Hugging Face model name (e.g., microsoft/DialoGPT-medium)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="specific data type(s) to calculate (comma-separated for multiple, e.g., fp16,bf16, default: use recommended dtype)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size for activation memory estimation (default: 1)"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="sequence length for activation memory estimation (default: 2048)"
    )
    
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA rank for fine-tuning memory estimation (default: 64)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show detailed parallelization strategies and recommendations"
    )
    
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="list all available data types and GPU types"
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="path to custom data_types.json, gpu_types.json, and display_settings.json (default: use config in this repo)"
    )
    
    parser.add_argument(
        "--local-config",
        type=str,
        default=None,
        help="path to local model config.json file instead of fetching from Hugging Face"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show program's version number and exit"
    )

    args = parser.parse_args()
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager(args.config_dir)
        # Initialize VRAM calculator with config
        vram_calc = VRAMCalculator(config_manager)
        
        # If user wants to list types, show and exit
        if args.list_types:
            config_manager.list_data_types()
            config_manager.list_gpu_types()
            return
        
        # Check if model name is provided
        if not args.model_name:
            console.print("[bold red]❌ Error:[/bold red] model_name is required unless using --list-types")
            parser.print_help()
            sys.exit(1)

        # Fetch config
        console.print(f"🔍 Fetching configuration for {args.model_name}...")
        config_path = ConfigParser.fetch_config(args.model_name, args.local_config)

        # Parse config
        console.print("📋 Parsing model configuration...")
        config = ConfigParser.parse_config(config_path, args.model_name)

        # Calculate parameters
        console.print("🧮 Calculating model parameters...")
        num_params = ParameterCalculator.calculate_transformer_params(config)

        # Calculate memory requirements
        console.print("💾 Computing memory requirements...")
        available_dtypes = list(config_manager.get_data_types().keys())

        # Determine which data types to calculate
        if args.dtype:
            # User specified dtype(s) - support both single dtype and comma-separated list
            if ',' in args.dtype:
                # Multiple dtypes specified (comma-separated)
                dtypes_to_calculate = [dtype.strip() for dtype in args.dtype.split(',')]
            else:
                # Single dtype specified
                dtypes_to_calculate = [args.dtype]

            # Validate all specified dtypes are available
            invalid_dtypes = [dtype for dtype in dtypes_to_calculate if dtype not in available_dtypes]
            if invalid_dtypes:
                console.print(f"[bold red]❌ Error:[/bold red] Invalid data types: {', '.join(invalid_dtypes)}")
                console.print(f"[dim]Available data types: {', '.join(available_dtypes)}[/dim]")
                sys.exit(1)
        else:
            # No dtype specified - use recommended dtype if available, otherwise smart fallback
            if config.recommended_dtype and config.recommended_dtype in available_dtypes:
                # Use recommended dtype only
                dtypes_to_calculate = [config.recommended_dtype]
                console.print(f"[dim]Using recommended data type: {config.recommended_dtype.upper()}[/dim]")
                console.print(f"[dim]Use --dtype to specify different type, or see --list-types for all options[/dim]")
            else:
                # Recommended dtype not available - smart fallback to fp16/bf16/fp32
                fallback_dtype = None
                for preferred in ['fp16', 'bf16', 'fp32']:
                    if preferred in available_dtypes:
                        fallback_dtype = preferred
                        break
                if fallback_dtype:
                    dtypes_to_calculate = [fallback_dtype]
                    if config.recommended_dtype:
                        console.print(f"[yellow]⚠️  Recommended dtype '{config.recommended_dtype}' not available[/yellow]")
                    console.print(f"[dim]Using default data type: {fallback_dtype.upper()} (fp16/bf16 preferred)[/dim]")
                    console.print(f"[dim]Use --dtype to specify different type, or see --list-types for all options[/dim]")
                else:
                    # No preferred types available, show all types
                    dtypes_to_calculate = available_dtypes
                    console.print(f"[yellow]⚠️  No preferred data types (fp16/bf16/fp32) available, showing all types[/yellow]")
        # Calculate all memory values at once using LlmodelMemoryResult
        memory_results = []
        for dtype in dtypes_to_calculate:
            memory_result = LlmodelMemoryResult(
                dtype=dtype,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                lora_rank=args.lora_rank
            )
            memory_result.calculate_all(config, num_params, vram_calc)
            memory_results.append(memory_result)
        # Print results
        print_results(config, memory_results, verbose=args.verbose)
        
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        sys.exit(1)
    finally:
        # Clean up global cache directory
        ConfigParser.cleanup_global_cache()


if __name__ == "__main__":
    main()
