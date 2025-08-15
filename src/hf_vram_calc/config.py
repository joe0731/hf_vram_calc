"""
Configuration Manager for VRAM Calculator

Handles loading and validation of separate configuration files for data types, GPU specifications, and display settings.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console

# create console instance for configuration messages
console = Console()


class ConfigManager:
    """Manage configuration loading and validation from separate config files"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else self._get_default_config_dir()
        self.data_types = self._load_data_types()
        self.gpu_types = self._load_gpu_types()
        self.display_settings = self._load_display_settings()
    
    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory path"""
        return Path(__file__).parent
    
    def _load_json_file(self, filename: str, default_content: Dict = None) -> Dict:
        """Load JSON file with error handling"""
        file_path = self.config_dir / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            if default_content is not None:
                console.print(f"[yellow]💡 Creating default {filename}...[/yellow]")
                self._create_default_file(filename, default_content)
                return default_content
            else:
                console.print(f"[bold red]❌ Configuration file not found:[/bold red] {file_path}")
                sys.exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]❌ Invalid JSON in {filename}:[/bold red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]❌ Error loading {filename}:[/bold red] {e}")
            sys.exit(1)
    
    def _create_default_file(self, filename: str, content: Dict):
        """Create default configuration file"""
        file_path = self.config_dir / filename
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
            console.print(f"[green]✓ Created default {filename}:[/green] {file_path}")
        except Exception as e:
            console.print(f"[bold red]❌ Could not create {filename}:[/bold red] {e}")
            sys.exit(1)
    
    def _load_data_types(self) -> Dict:
        """Load data types configuration"""
        default_data_types = {
            "fp32": {"bytes_per_param": 4, "description": "32-bit floating point"},
            "fp16": {"bytes_per_param": 2, "description": "16-bit floating point"},
            "bf16": {"bytes_per_param": 2, "description": "Brain Float 16"},
            "int8": {"bytes_per_param": 1, "description": "8-bit integer"},
            "int4": {"bytes_per_param": 0.5, "description": "4-bit integer"}
        }
        
        data_types = self._load_json_file("data_types.json", default_data_types)
        
        # validate data types
        for dtype, info in data_types.items():
            if "bytes_per_param" not in info:
                raise ValueError(f"missing 'bytes_per_param' for data type: {dtype}")
        
        return data_types
    
    def _load_gpu_types(self) -> List[Dict]:
        """Load GPU types configuration"""
        default_gpu_types = [
            {"name": "RTX 4090", "memory_gb": 24, "category": "consumer"},
            {"name": "A100 80GB", "memory_gb": 80, "category": "datacenter"}
        ]
        
        gpu_types = self._load_json_file("gpu_types.json", default_gpu_types)
        
        # validate GPU types
        if not isinstance(gpu_types, list):
            raise ValueError("gpu_types.json must contain a list of GPU configurations")
        
        for gpu in gpu_types:
            if "name" not in gpu or "memory_gb" not in gpu:
                raise ValueError(f"invalid GPU configuration: {gpu}")
        
        return gpu_types
    
    def _load_display_settings(self) -> Dict:
        """Load display settings configuration"""
        default_display_settings = {
            "max_gpu_display": 8,
            "preferred_categories": ["datacenter", "consumer"],
            "memory_color_thresholds": {"green": 8, "yellow": 24, "red": 80}
        }
        
        return self._load_json_file("display_settings.json", default_display_settings)
    
    def get_data_types(self) -> Dict[str, float]:
        """Get data types mapping"""
        return {
            dtype: info["bytes_per_param"] 
            for dtype, info in self.data_types.items()
        }
    
    def get_data_type_descriptions(self) -> Dict[str, str]:
        """Get data type descriptions"""
        return {
            dtype: info.get("description", "") 
            for dtype, info in self.data_types.items()
        }
    
    def get_gpu_types(self) -> List[Tuple[str, int, str]]:
        """Get GPU types list with category"""
        return [
            (gpu["name"], gpu["memory_gb"], gpu.get("category", "unknown")) 
            for gpu in self.gpu_types
        ]
    
    def get_display_settings(self) -> Dict:
        """Get display settings"""
        return self.display_settings
    
    def add_data_type(self, name: str, bytes_per_param: float, description: str = ""):
        """Add new data type to configuration"""
        self.data_types[name] = {
            "bytes_per_param": bytes_per_param,
            "description": description
        }
        self._save_data_types()
    
    def add_gpu_type(self, name: str, memory_gb: int, category: str = "custom", architecture: str = ""):
        """Add new GPU type to configuration"""
        new_gpu = {
            "name": name,
            "memory_gb": memory_gb,
            "category": category
        }
        if architecture:
            new_gpu["architecture"] = architecture
        
        self.gpu_types.append(new_gpu)
        self._save_gpu_types()
    
    def _save_data_types(self):
        """Save data types configuration to file"""
        file_path = self.config_dir / "data_types.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data_types, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[bold red]❌ Could not save data_types.json:[/bold red] {e}")
    
    def _save_gpu_types(self):
        """Save GPU types configuration to file"""
        file_path = self.config_dir / "gpu_types.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.gpu_types, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[bold red]❌ Could not save gpu_types.json:[/bold red] {e}")
    
    def _save_display_settings(self):
        """Save display settings configuration to file"""
        file_path = self.config_dir / "display_settings.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.display_settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[bold red]❌ Could not save display_settings.json:[/bold red] {e}")
    
    def list_data_types(self):
        """List all available data types"""
        console.print("\n[bold cyan]Available Data Types:[/bold cyan]")
        
        from rich.table import Table
        from rich import box
        
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Data Type", style="cyan")
        table.add_column("Bytes/Param", style="green", justify="right")
        table.add_column("Description", style="white")
        
        for dtype, info in self.data_types.items():
            table.add_row(
                dtype.upper(),
                str(info["bytes_per_param"]),
                info.get("description", "")
            )
        
        console.print(table)
    
    def list_gpu_types(self):
        """List all available GPU types"""
        console.print("\n[bold cyan]Available GPU Types:[/bold cyan]")
        
        from rich.table import Table
        from rich import box
        
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("GPU Name", style="cyan")
        table.add_column("Memory (GB)", style="green", justify="right")
        table.add_column("Category", style="yellow")
        table.add_column("Architecture", style="white")
        
        for gpu in self.gpu_types:
            table.add_row(
                gpu["name"],
                str(gpu["memory_gb"]),
                gpu.get("category", "unknown"),
                gpu.get("architecture", "N/A")
            )
        
        console.print(table)


def load_config(config_dir: Optional[str] = None) -> ConfigManager:
    """Convenience function to load configuration"""
    return ConfigManager(config_dir)
