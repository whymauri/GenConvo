## GenConvo

Replication of GenConvoBench (see `https://arxiv.org/pdf/2506.06266`) using Verdict to synthesize question/answer datasets from long documents.

### Installation (uv)

- Requires Python 3.12+
- Using `uv` from the repo root:

```bash
# Ensure Python 3.12 is available to uv
uv python install 3.12

# Create a local virtual env and install project deps
uv venv -p 3.12
uv sync --group dev

# Verify CLI (no need to activate)
uv run genconvo -h | cat
```

If your provider requires credentials (e.g., Anthropic), configure them as usual before running.

### Quickstart (FinanceBench warmup)

The CLI is currently hardcoded to FinanceBench. Pass a FinanceBench `doc_name` (without extension), e.g., `AMD_2022_10K`.

Warmup run (factual prompt; 1 question; 1 worker; model `claude-sonnet-4-20250514`; temperature 0.7):

```bash
genconvo AMD_2022_10K --warmup --print-json
```

Notes:

- `--warmup` overrides `--num-questions`, `--max-workers`, `--model-name`, and `--temperature` (a warning is printed).
- The CLI expects a file named `<doc_name>.md` under `FINANCE_BENCH_PATH` (see `src/genconvo/data/finance/__init__.py`).

### Full run (16 questions, parallel)

Run with more questions and workers:

```bash
genconvo AMD_2022_10K \
  --num-questions 16 \
  --max-workers 16 \
  --model-name claude-sonnet-4-20250514 \
  --temperature 0.7 \
  --print-json
```

### Where results are saved

Each run saves Q&A pairs as a HuggingFace dataset directory under `data/genconvo/`.

Load and inspect in Python:

```python
from datasets import Dataset
from pathlib import Path

root = Path("data/genconvo")
latest = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime)[-1]
ds = Dataset.load_from_disk(str(latest))

print(ds)
print(ds[0]["question"])  # first question
print(ds[0]["answer"])    # first answer
```

### Preparing FinanceBench markdowns (optional)

If the `<doc_name>.md` file is not present under `FINANCE_BENCH_PATH`, you can generate markdowns from PDFs using the helper in `src/genconvo/data/finance/load.py`:

```python
from pathlib import Path
from genconvo.data.finance.load import load_finance

# e.g., only the AMD 10-K
df = load_finance(doc_names=["AMD_2022_10K"], force=False)
print("Saved to:", df.head())
```

This will create markdown files alongside the HuggingFace cache for FinanceBench. We have an example
in `notebooks/test_finance_bench.ipynb`.

### Defaults

- Prompt type: `factual`
- Model: `claude-sonnet-4-20250514` (override with `--model-name`)
- Temperature: `0.7` (override with `--temperature`)

You can tune `--num-questions` and `--max-workers` to scale generation.
