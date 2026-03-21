# Text Coach

An AI-powered app that helps you craft emotionally intelligent replies to messages you receive. Paste in a message from a friend, and the app suggests a thoughtful reply — powered by a fine-tuned Qwen2.5-1.5B model.

---

## One-time Setup

```bash
# 1. Clone the repo and enter the project folder
git clone <your-repo-url>
cd chatbot

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Create your .env file
cp .env.example .env
# Open .env and fill in your API keys
```

Install Ollama (if you want to run the fine-tuned model locally): https://ollama.com

---

## Ollama Setup

Run this **once** after you have your GGUF file on HuggingFace.

**Step 1 — Download your GGUF file**

1. Go to your HuggingFace repo (e.g. `https://huggingface.co/yourname/qwen-emotional-coach-v1-gguf`)
2. Click the `.gguf` file and download it
3. Move it into the `ollama/` folder in this project:

```
chatbot/
└── ollama/
    ├── Modelfile
    └── qwen-emotional-coach-Q4_K_M.gguf   ← place it here
```

**Step 2 — Register the model with Ollama**

From the `chatbot/` project root:

```bash
ollama create qwen-emotional-coach -f ollama/Modelfile
```

Verify it worked:

```bash
ollama list
# should show: qwen-emotional-coach
```

Quick smoke test (optional):

```bash
ollama run qwen-emotional-coach "hello"
```

**Step 3 — Update your .env**

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen-emotional-coach
```

---

## Starting the App

Pick the path that matches your `.env`, then open the required terminals.

---

### Path 1 — OpenRouter (cloud API, default for dev)

**What it is:** Uses a cloud model (GPT-4o-mini or similar) via OpenRouter. No GPU or local model needed. Use this when you want to test the app quickly without running anything locally.

`.env`:
```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-...
MODEL_NAME=openai/gpt-4o-mini
```

Open **2 terminals:**

```bash
# Terminal 1 — FastAPI backend
source .venv/bin/activate
uvicorn api.main:app --reload
```

```bash
# Terminal 2 — Streamlit frontend
source .venv/bin/activate
streamlit run frontend/app.py
```

---

### Path 2 — Ollama (local fine-tuned model)

**What it is:** Runs your fine-tuned Qwen GGUF model entirely on your Mac — no internet needed after setup. Use this to test your trained model locally before deploying to production.

Prerequisite: complete the **Ollama Setup** section above first.

`.env`:
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen-emotional-coach
```

Open **3 terminals:**

```bash
# Terminal 1 — Ollama server (the local LLM engine)
ollama serve
```

```bash
# Terminal 2 — FastAPI backend
source .venv/bin/activate
uvicorn api.main:app --reload
```

```bash
# Terminal 3 — Streamlit frontend
source .venv/bin/activate
streamlit run frontend/app.py
```

---

### Path 3 — vLLM on RunPod (production)

**What it is:** Your fine-tuned model running on a cloud GPU (RunPod). The app code is identical to the other paths — it just points at a remote GPU server instead of your Mac. Use this once you're ready to serve real users.

Prerequisite: a RunPod instance must already be running with vLLM serving your merged HuggingFace model.

`.env`:
```bash
LLM_PROVIDER=vllm
VLLM_BASE_URL=https://YOUR-POD-ID-8000.proxy.runpod.net/v1
VLLM_API_KEY=your-vllm-token
VLLM_MODEL=yourname/qwen-emotional-coach-v1-merged
```

Open **2 terminals** (vLLM runs remotely — nothing to start locally):

```bash
# Terminal 1 — FastAPI backend
source .venv/bin/activate
uvicorn api.main:app --reload
```

```bash
# Terminal 2 — Streamlit frontend
source .venv/bin/activate
streamlit run frontend/app.py
```

---

Then open http://localhost:8501 in your browser. Paste a message, click **Send as Friend**, and the app returns a reply from whichever model is active.

---

## Switching LLM Backends

Change `LLM_PROVIDER` in `.env` — no code changes needed:

| Provider | Use case | `.env` setting |
|----------|----------|----------------|
| `openrouter` | Cloud API, any model (default dev) | `LLM_PROVIDER=openrouter` |
| `ollama` | Local fine-tuned model via Ollama | `LLM_PROVIDER=ollama` |
| `vllm` | Production — fine-tuned model on RunPod | `LLM_PROVIDER=vllm` |

---

## Project Structure

```
api/          FastAPI backend — routes, LLM calls, SQLite history
frontend/     Streamlit UI
data_pipeline/ Screenshot OCR → CSV/JSONL training data
training/     Fine-tuning scripts and configs (run on Google Colab)
ollama/       Modelfile for registering the GGUF with Ollama
```

See `CLAUDE.md` for the full architecture, deployment guide, and training workflow.
