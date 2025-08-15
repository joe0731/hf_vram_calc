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
from .calculator import VRAMCalculator, ParameterCalculator
from .parallel import ParallelizationCalculator
from .models import ModelConfig

# Create global console instance
console = Console()


def format_memory_size(memory_gb: float) -> str:
    """Format memory size with appropriate unit"""
    if memory_gb >= 1024:
        return f"{memory_gb / 1024:.2f} TB"
    else:
        return f"{memory_gb:.2f} GB"


def print_memory_table(results: Dict, num_params: int):
    """Print memory requirements table by data type and scenario"""
    console.print()
    
    # Create beautiful table
    table = Table(
        title="💾 Memory Requirements by Data Type and Scenario",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold blue",
        show_lines=True
    )
    
    # Add columns
    table.add_column("Data Type", justify="center", style="cyan", width=12)
    table.add_column("Total Size\n(GB)", justify="right", style="green", width=12)
    table.add_column("Inference\n(GB)", justify="right", style="yellow", width=12)
    table.add_column("Training\n(Adam) (GB)", justify="right", style="red", width=15)
    table.add_column("LoRA\n(GB)", justify="right", style="blue", width=12)
    
    # Add rows - display all available data types from results
    for dtype in sorted(results['memory_by_dtype'].keys()):
        base_memory = results['memory_by_dtype'][dtype]
        inference_memory = VRAMCalculator.calculate_inference_memory(base_memory)
        training_memory = VRAMCalculator.calculate_training_memory(base_memory)
        lora_rank = results.get("lora_rank", 64)
        lora_memory = VRAMCalculator.calculate_lora_memory(base_memory, num_params, lora_rank)
        
        # Format numbers with appropriate colors
        base_str = f"{base_memory:.2f}"
        inference_str = f"{inference_memory:.2f}"
        training_str = f"{training_memory:.2f}"
        lora_str = f"{lora_memory:.2f}"
        
        table.add_row(
            dtype.upper(),
            base_str,
            inference_str,
            training_str,
            lora_str
        )
    
    console.print(table)


def print_parallelization_table(results: Dict):
    """Print parallelization strategies table"""
    console.print()
    
    # Use the first available data type for parallelization calculations
    available_dtypes = list(results['memory_by_dtype'].keys())
    if not available_dtypes:
        return
    
    # Prefer bf16 or fp16, otherwise use the first available
    preferred_dtype = None
    for dtype in ['bf16', 'fp16', 'fp32']:
        if dtype in available_dtypes:
            preferred_dtype = dtype
            break
    
    if preferred_dtype is None:
        preferred_dtype = available_dtypes[0]
    
    base_memory = results['memory_by_dtype'][preferred_dtype]
    inference_memory = VRAMCalculator.calculate_inference_memory(base_memory)
    
    # Create beautiful parallelization table
    table = Table(
        title=f"⚡ Parallelization Strategies ({preferred_dtype.upper()} Inference)",
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


def print_detailed_recommendations(results: Dict, config: ModelConfig):
    """Print detailed recommendations"""
    # Use the first available data type for recommendations
    available_dtypes = list(results['memory_by_dtype'].keys())
    if not available_dtypes:
        return
    
    # Prefer bf16 or fp16, otherwise use the first available
    preferred_dtype = None
    for dtype in ['bf16', 'fp16', 'fp32']:
        if dtype in available_dtypes:
            preferred_dtype = dtype
            break
    
    if preferred_dtype is None:
        preferred_dtype = available_dtypes[0]
    
    base_memory = results['memory_by_dtype'][preferred_dtype]
    inference_memory = VRAMCalculator.calculate_inference_memory(base_memory)
    training_memory = VRAMCalculator.calculate_training_memory(base_memory)
    num_params = ParameterCalculator.calculate_transformer_params(config)
    lora_memory = VRAMCalculator.calculate_lora_memory(base_memory, num_params)
    
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
    max_display = display_settings.get("max_gpu_display", 8)
    
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


def print_model_header(config: ModelConfig, num_params: int):
    """Print beautiful model information header"""
    console.print()
    
    # Create model info panel
    model_info = f"""
[bold]Model:[/bold] [cyan]{config.model_name}[/cyan]
[bold]Architecture:[/bold] [magenta]{config.model_type}[/magenta]
[bold]Parameters:[/bold] [green]{num_params:,}[/green]
    """
    
    header_panel = Panel(
        model_info.strip(),
        title="🤖 Model Information",
        border_style="bright_cyan",
        padding=(1, 2),
        expand=False
    )
    
    console.print(Align.center(header_panel))


def print_results(config: ModelConfig, num_params: int, results: Dict):
    """Print comprehensive formatted results"""
    # Print model header
    print_model_header(config, num_params)
    
    # Print main memory table
    print_memory_table(results, num_params)
    
    # Print parallelization table
    print_parallelization_table(results)
    
    # Print detailed recommendations
    print_detailed_recommendations(results, config)


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
        """
    )
    
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Hugging Face model name (e.g., microsoft/DialoGPT-medium)"
    )
    
    parser.add_argument(
        "--dtype",
        choices=available_dtypes,
        default=None,
        help="specific data type to calculate (default: show all)"
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
        "--show-detailed",
        action="store_true",
        default=True,
        help="show detailed parallelization strategies and recommendations"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="path to configuration directory (default: same as script)"
    )
    
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="list all available data types and GPU types"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager(args.config_dir)
        
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
        
        # Initialize VRAM calculator with config
        vram_calc = VRAMCalculator(config_manager)
        
        # Use rich progress for better UX
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Fetch and parse config
            task1 = progress.add_task(f"🔍 Fetching configuration for {args.model_name}...", total=100)
            config_data = ConfigParser.fetch_config(args.model_name)
            progress.update(task1, completed=100)
            
            # Parse config
            task2 = progress.add_task("📋 Parsing model configuration...", total=100)
            config = ConfigParser.parse_config(config_data, args.model_name)
            progress.update(task2, completed=100)
            
            # Calculate parameters
            task3 = progress.add_task("🧮 Calculating model parameters...", total=100)
            num_params = ParameterCalculator.calculate_transformer_params(config)
            progress.update(task3, completed=100)
            
            # Calculate memory requirements
            task4 = progress.add_task("💾 Computing memory requirements...", total=100)
            results = {"memory_by_dtype": {}}
            
            available_dtypes = list(config_manager.get_data_types().keys())
            dtypes_to_calculate = [args.dtype] if args.dtype else available_dtypes
            
            for dtype in dtypes_to_calculate:
                memory_gb = vram_calc.calculate_model_memory(num_params, dtype)
                results["memory_by_dtype"][dtype] = memory_gb
            
            # Store LoRA rank for calculations
            results["lora_rank"] = args.lora_rank
            progress.update(task4, completed=100)
        
        # Print results
        if args.show_detailed:
            print_results(config, num_params, results)
        else:
            # Simplified output - just the header and main table
            print_model_header(config, num_params)
            print_memory_table(results, num_params)
        
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
