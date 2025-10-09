"""
Minimal decider: use model_size_gb from hf-vram-calc JSON (or provided file),
compare to GPU VRAM with simple per-GPU ratio. No complex rules or exceptions.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal decider with two modes: "
            "(1) --hf_json <file> (no model/dtype needed); "
            "(2) --model <id> [--dtype <type>] (dtype optional)"
        ),
    )
    parser.add_argument("--gpu_model", required=True, type=str, help="GPU model key in exps/gpu_models.json (e.g., L40S)")
    parser.add_argument("--num_gpus", required=True, type=int, help="Number of GPUs available")
    parser.add_argument("--model", required=False, type=str, help="Hugging Face model id (required if --hf_json is not provided)")
    parser.add_argument("--dtype", default=None, type=str, help="Optional dtype to pass to hf-vram-calc (e.g., bf16, fp8)")
    parser.add_argument("--gpu_models_path", default=str(Path(__file__).resolve().parents[1] / "exps" / "gpu_models.json"), type=str, help="Path to gpu models caps JSON")
    parser.add_argument("--hf_json", default=None, type=str, help="Path to precomputed hf-vram-calc JSON; if absent, will run CLI")
    parser.add_argument("--output_json", default=None, type=str, help="Optional path to write decision JSON")
    return parser.parse_args()


def load_gpu_models(models_path: Path) -> Dict[str, Any]:
    with models_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_hf_vram_calc(model: str, dtype: str | None) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_json = Path(tmpdir) / "hf_vram_calc_out.json"
        cmd: List[str] = ["hf-vram-calc", "--model", model, "--output_json", str(tmp_json)]
        if dtype:
            cmd.extend(["--dtype", dtype])
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(tmp_json.read_text(encoding="utf-8"))


def read_hf_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_memory_entry(memory_requirements: List[Dict[str, Any]], dtype: str | None) -> Dict[str, Any]:
    target = (dtype or "").upper()
    for entry in memory_requirements:
        if entry.get("dtype") == target:
            return entry
    return memory_requirements[0]


def prepare_from_hf_json(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Prepare inputs and memory selection when an hf-vram-calc JSON is provided."""
    hf_json = read_hf_json(Path(args.hf_json))
    mem_entry = hf_json["memory_requirements"][0]
    inputs: Dict[str, Any] = {
        "gpu_model": args.gpu_model,
        "num_gpus": int(args.num_gpus),
        "model": hf_json.get("model", {}).get("name"),
        "dtype": mem_entry.get("dtype"),
    }
    return inputs, mem_entry, hf_json


def prepare_from_cli(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Prepare inputs and memory selection by running hf-vram-calc from CLI args."""
    hf_json = run_hf_vram_calc(args.model, args.dtype)
    mem_entry = pick_memory_entry(hf_json["memory_requirements"], args.dtype)
    inputs: Dict[str, Any] = {
        "gpu_model": args.gpu_model,
        "num_gpus": int(args.num_gpus),
        "model": args.model,
        "dtype": mem_entry.get("dtype") if not args.dtype else args.dtype,
    }
    # If dtype not passed, reflect selected entry's dtype
    if not args.dtype:
        inputs["dtype"] = mem_entry.get("dtype")
    return inputs, mem_entry, hf_json


def compute_calc(mem_entry: Dict[str, Any], vram_gb: float, num_gpus: int) -> Dict[str, Any]:
    model_size_gb = float(mem_entry["model_size_gb"])  # assume present
    per_gpu_required = model_size_gb / max(1, int(num_gpus))
    ratio = per_gpu_required / float(vram_gb)
    return {
        "memory_estimate_gb": round(model_size_gb, 2),
        "gpu_vram_gb": float(vram_gb),
        "per_gpu_required_gb": round(per_gpu_required, 2),
        "ratio": round(ratio, 4),
        "dtype_used": mem_entry.get("dtype"),
    }


def decide(ratio: float) -> str:
    return "RUN" if ratio <= 1.0 else "SKIP"


def emit(out: Dict[str, Any], output_json: str | None) -> None:
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if output_json:
        Path(output_json).write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    args = parse_args()

    # GPU caps (assume valid key)
    gpu_models = load_gpu_models(Path(args.gpu_models_path))
    gpu_caps = gpu_models[args.gpu_model]

    # Branch BEFORE preparing inputs
    hf_path = Path(args.hf_json) if args.hf_json else None
    if hf_path and hf_path.exists():
        inputs, mem_entry, hf_json = prepare_from_hf_json(args)
    else:
        if not args.model:
            emit({"error": "--model is required when --hf_json is not provided"}, args.output_json)
            return
        inputs, mem_entry, hf_json = prepare_from_cli(args)

    # Compute
    calc = compute_calc(mem_entry, float(gpu_caps["vram_gb"]), inputs["num_gpus"])
    decision = decide(calc["ratio"])

    out = {"decision": decision, "inputs": inputs, "calc": calc}
    emit(out, args.output_json)


if __name__ == "__main__":
    main()


