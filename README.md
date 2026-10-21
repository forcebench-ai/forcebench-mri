# forcebench-mri

Looking inside an open-weights model while it does Salesforce work: logit lens, linear probes,
activation patching, head attribution and steering, on [Forcebench](https://forcebench.ai) tasks.

Work in progress. Runs with [MLX](https://github.com/ml-explore/mlx) on Apple silicon.

```bash
uv sync
uv run pytest                      # hooks on a tiny random model, no downloads
MRI_MODEL_PATH=<model dir> uv run python scripts/smoke.py --prompts <prompts.json> --out outputs/smoke
```
