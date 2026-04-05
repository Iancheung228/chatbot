# CLAUDE.md — Project Context for Claude Code

## What This Project Does

**Text Coach** — an AI-powered app that helps users craft emotionally intelligent replies to messages they receive. The user is practising ML engineering with a goal of deploying this as a production-scale internet app. Fine-tuned Qwen2.5-1.5B replaces the cloud LLM backend in production.

Three-layer architecture:
1. **Serving layer** — FastAPI backend + Streamlit frontend
2. **Data pipeline** — WeChat screenshot → PaddleOCR → CSV → JSONL fine-tuning data
3. **Training** — Qwen2.5-1.5B fine-tuned with Unsloth on Google Colab GPU (notebook uploaded manually, not via git pull)

---

## How to Run (Local Dev — nothing changes)

```bash
# 1. Activate venv (always use .venv, not venv/ or app/venv/)
source .venv/bin/activate

# 2. API backend
uvicorn api.main:app --reload

# 3. Streamlit frontend (separate terminal)
streamlit run frontend/app.py

# 4. Data pipeline (after OCR images are in convos_folder76/ or similar)
python -m data_pipeline.process \
  --root_folder convos_folder76 \
  --out_csv output/all_messages.csv \
  --out_json output/all_conversations.json \
  --out_jsonl output/unsloth_chatml.jsonl

# 5. Fine-tuning (upload explore_v2.ipynb to Google Colab manually — no git pull)
#    - Upload training/explore_v2.ipynb to Colab
#    - Upload all_conversations_cleaned.json to Google Drive root
#    - Run cells top-to-bottom

# 6. Docker (optional — use to test production behaviour locally)
docker compose up --build
```

---

## Production Deployment Architecture

```
[User browser] → [Streamlit — Streamlit Community Cloud]
                        ↓ API_BASE_URL
               [FastAPI — Railway]
                        ↓ OpenAI-compatible HTTP
               [vLLM — RunPod GPU (~$0.20/hr, pay when ON)]
                        ↓ loads weights
               [HuggingFace Hub — merged fine-tuned model]
```

The `.env` file is the only switch between dev and production — no code changes:

```bash
# LOCAL DEV
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-...
MODEL_NAME=openai/gpt-4o-mini

# PRODUCTION (set as env vars on Railway + Streamlit Cloud)
LLM_PROVIDER=vllm
VLLM_BASE_URL=https://YOUR-RUNPOD.proxy.runpod.net/v1
VLLM_API_KEY=your-vllm-token
VLLM_MODEL=yourname/qwen-emotional-coach-v1-merged
API_BASE_URL=https://your-app.railway.app   # REQUIRED on Streamlit Cloud
DB_PATH=/data/chatbot.db
```

---

## Directory Map

```
chatbot/
├── api/                  FastAPI backend — routes, LLM calls, SQLite history
├── frontend/             Streamlit UI (run with streamlit run frontend/app.py)
├── data_pipeline/        Screenshot OCR → CSV/JSONL (runs locally or on Colab)
├── training/
│   ├── configs/          One YAML per experiment (qwen_v1.yaml = baseline)
│   └── prompts/          System prompts as text files (git-tracked, version-controlled)
├── tests/                Pytest tests
├── Dockerfile            API container (for Railway / cloud deploy)
├── Dockerfile.frontend   Streamlit container
├── docker-compose.yml    Runs both containers locally for production testing
├── output/               GITIGNORED — generated CSV/JSON/JSONL from data pipeline
├── lora_outputs/         GITIGNORED — LoRA adapter weights from training runs
└── convos_folder76/      GITIGNORED — raw WeChat screenshot images (source data)
```

---

## Key Files

| File | Role |
|------|------|
| `api/config.py` | All settings via pydantic-settings. Single source of truth for env vars. |
| `api/llm.py` | All LLM backends (OpenRouter, vLLM, Ollama) + `build_api_payload()` + streaming |
| `api/db.py` | SQLite conversation history (save_message, log_llm_suggestion, get_last_messages) |
| `data_pipeline/ocr.py` | PaddleOCR helpers: `get_ocr_engine`, `run_ocr_with_cache`, `parse_paddle_result_dict` |
| `data_pipeline/process.py` | Pipeline orchestration: `process_all_root`, `reconcile_rows`, `to_jsonl` |
| `training/configs/qwen_v1.yaml` | Baseline experiment config — copy to start a new experiment |
| `training/prompts/system_v1.txt` | Chinese system prompt for fine-tuning (edit here, not in code) |

---

## Import Conventions

Always use absolute imports from project root. Never use relative imports.

```python
from api.config import settings          # correct
from api.db import save_message          # correct
from data_pipeline.ocr import get_ocr_engine  # correct

from .config import settings             # WRONG — relative import
```

`frontend/app.py` adds the project root to `sys.path` at the top — this is intentional and necessary because Streamlit adds the script's own directory to `sys.path`, not the project root.

---

## Environment

- **Active venv**: `.venv/` — always use this, never `venv/` (old, deleted)
- **Env file**: `.env` (gitignored) — copy from `.env.example` and fill in keys
- **Required keys**: `OPENROUTER_API_KEY` (dev) or `BASE_URL` + `OPENROUTER_API_KEY` (prod vLLM)

---

## Pluggable LLM Backend

Switch between any OpenAI-compatible LLM by changing `.env` only:

```bash
# Dev — OpenRouter (cheap, no GPU)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-...
MODEL_NAME=openai/gpt-4o-mini           # or meta-llama/llama-3-8b-instruct, etc.

# Prod — vLLM serving fine-tuned Qwen on RunPod
LLM_PROVIDER=vllm
VLLM_BASE_URL=https://YOUR-RUNPOD.proxy.runpod.net/v1
VLLM_API_KEY=your-vllm-token
VLLM_MODEL=yourname/qwen-emotional-coach-v1-merged

# Local — Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen-emotional-coach
```

No code changes required — `api/llm.py` uses `settings.model_name` and `settings.base_url`.

---

## Experiment Workflow (Training)

One experiment = one YAML config file + one WandB run + one HF Hub model version, all sharing the same `run_name`.

```
1. Edit code or YAML in VSCode (with Claude Code's help)
2. Upload training/explore_v2.ipynb to Google Colab manually
3. Upload all_conversations_cleaned.json to Google Drive root (for IN_JSON path)
4. Run cells top-to-bottom on Colab GPU runtime
5. Adapter → HF Hub (HF_HUB_REPO)
6. Merged model → HF Hub (HF_HUB_REPO + "-merged")  ← use this in production MODEL_NAME
```

For a new experiment: copy `qwen_v1.yaml` → `qwen_v2.yaml`, change what you're testing, commit.

---

## What Is and Isn't in Git

| In git | Not in git (why) |
|--------|-----------------|
| All source code | `.env` (secrets) |
| `training/configs/*.yaml` | `output/` (generated) |
| `training/prompts/*.txt` | `lora_outputs/` (large binary weights → HF Hub) |
| `requirements*.txt` | `convos_folder76/` (raw image data, large) |
| `Dockerfile`, `docker-compose.yml` | `*.csv`, `*.db`, `*.zip` |
| `.env.example` | |

---

## Prompt Caching

LLM backends cache the KV-states for tokens they've already computed. When consecutive requests share an identical prefix, those tokens are not re-processed — saving latency and (on paid APIs) cost. The cacheable prefix must be **byte-for-byte identical** up to the cache boundary.

### How each backend handles it

| Backend | Mechanism | Action required |
|---------|-----------|-----------------|
| **Ollama** (local dev) | Automatic llama.cpp prefix caching | None — works out of the box |
| **vLLM** (RunPod, prod) | Automatic prefix caching | None — enabled by default |
| **OpenRouter → Anthropic** | Explicit opt-in | Add `"cache_control": {"type": "ephemeral"}` to the message (not yet implemented) |

### Why `{SUMMARY}` must stay at the END of the system prompt

The system prompt in `training/prompts/system_v3.txt` is structured as:

```
[static: Role, Protocol, Rules, Examples, Core Identity]  ← ~600 tokens, fully cacheable
---
# Conversation Context
{SUMMARY}                                                  ← dynamic, per-conversation
```

Any token after a change breaks the cache. Keeping the static body first means ~600 tokens are cached after the first request. If `{SUMMARY}` were moved back toward the top (as in v3's original position), only ~80 tokens would be cacheable.

**Do not move `{SUMMARY}` to the middle or top of the system prompt** when editing future prompt versions.

---

## Requirements Files

| File | Install when |
|------|-------------|
| `requirements.txt` | Always — API + frontend |
| `requirements-data.txt` | Running the OCR data pipeline |
| `requirements-training.txt` | Running fine-tuning (Colab — install via !pip install -r) |
| `requirements-dev.txt` | Running tests / linting |
