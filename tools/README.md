## Decider: quick GPU fit check for LLMs

`decider.py` answers: Can this GPU setup run my model? It compares the model's weight memory (`model_size_gb`) to the per‑GPU VRAM capacity and returns RUN or SKIP.

### How it works (simple rule)
- Per‑GPU requirement = `model_size_gb / num_gpus`
- Decision: RUN if per‑GPU requirement ≤ GPU VRAM; otherwise SKIP

### Inputs
- `gpu_model`: key from `exps/gpu_models.json` (provides VRAM per GPU)
- `num_gpus`: integer number of GPUs
- Model info comes from either:
  - A precomputed JSON from `hf-vram-calc` (recommended), or
  - Running `hf-vram-calc` on the fly (dtype optional)

### Recommended workflow (faster, test many GPUs)
1) Generate model memory once with `hf-vram-calc`:

```bash
hf-vram-calc --model google/gemma-3-27b-it --dtype fp8 --output_json exps/gemma.json
```

2) Run the decider against different GPUs without recalculating the model:

```bash
python tools/decider.py --gpu_model L40S --num_gpus 2 --hf_json exps/gemma.json
python tools/decider.py --gpu_model H100 --num_gpus 2 --hf_json exps/gemma.json
python tools/decider.py --gpu_model B200 --num_gpus 1 --hf_json exps/gemma.json
```

### Alternative workflow (calculate on the fly)
If you don't have a JSON yet, the decider can call `hf-vram-calc` for you. `--dtype` is optional; if omitted, `hf-vram-calc` uses its default/recommended dtype.

```bash
python tools/decider.py --gpu_model L40S --num_gpus 2 --model google/gemma-3-27b-it
# or specify dtype explicitly
python tools/decider.py --gpu_model L40S --num_gpus 2 --model google/gemma-3-27b-it --dtype fp8
```

### Output (JSON)
```json
{
  "decision": "RUN",
  "inputs": {
    "gpu_model": "L40S",
    "num_gpus": 2,
    "model": "google/gemma-3-27b-it",
    "dtype": "FP8"
  },
  "calc": {
    "memory_estimate_gb": 26.47,
    "gpu_vram_gb": 46.0,
    "per_gpu_required_gb": 13.23,
    "ratio": 0.2877,
    "dtype_used": "FP8"
  }
}
```

### Notes
- Only `memory_requirements -> model_size_gb` is used from the JSON.
- `gpu_models.json` defines available GPUs and their VRAM; edit `exps/gpu_models.json` to add or adjust entries.
- The decider is intentionally simple (weights-only, perfect sharding). It is a quick estimator, not a deployment guarantee.


