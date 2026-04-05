# Engineering Decisions Log

> This document records the challenges, trade-offs, and architectural decisions made while building the Text Coach project. It is written to be readable as a blog post — not just a changelog.

---

## The Project

The goal: build an AI app that helps a user craft emotionally intelligent replies to messages they receive, and eventually fine-tune a small language model to replicate a specific person's communication style from real WeChat conversations.

What started as a Streamlit script calling GPT-4o-mini has grown into a production-structured codebase with a data pipeline, training infrastructure, and a plan to serve a locally-hosted fine-tuned model. This document captures the engineering decisions made along the way.

---

## Decision 1: Restructure First, Build Second

**The situation:** The project worked — you could paste a message and get reply suggestions. But the codebase was a mess: OCR data processing code lived inside the FastAPI package, `requirement.txt` was misspelled and incomplete, there was no `.gitignore`, and two `venv/` directories were committed to git (thousands of binary files polluting the history).

**The temptation:** Just keep building. It works. Technical debt can wait.

**The decision:** Restructure first. Pay the debt before it compounds.

The reasoning: every new feature we'd add would import from `app.*`. Every data pipeline change would risk breaking the API. Every Colab session would pull thousands of venv files that serve no purpose. The longer we waited, the more painful the restructure would become.

**The outcome:**
```
Before                          After
──────────────────────────────  ──────────────────────────────
app/                            api/           (serving only)
  main.py                       data_pipeline/ (OCR + dataset)
  llm.py                        frontend/      (Streamlit)
  ocr.py  ← wrong place         training/      (fine-tuning)
  dataMain.py ← wrong place
frontend.py  ← at root
requirement.txt ← misspelled
(no .gitignore)
```

**The lesson:** Structural debt is the most expensive kind. It doesn't show up as a bug — it shows up as friction in every future task.

---

## Decision 2: Three Separate Dependency Trees

**The situation:** The project has three distinct operational contexts:
- The **API + frontend** runs all the time on a local machine
- The **data pipeline** (PaddleOCR) is heavy (~2GB of dependencies) and runs occasionally
- The **fine-tuning** (Unsloth, bitsandbytes) only runs on a GPU machine (Google Colab)

**The decision:** Three requirements files, not one.

```
requirements.txt          → API + frontend (always installed)
requirements-data.txt     → OCR pipeline (install when processing images)
requirements-training.txt → Fine-tuning (install on Colab only)
```

**Why this matters:** If you put PaddleOCR in `requirements.txt`, every deployment of the API drags in gigabytes of CV dependencies that the API never touches. If you put Unsloth in `requirements.txt`, every local dev environment needs CUDA-compatible packages that won't install on a Mac.

Each requirements file should reflect the machine and task it's meant for, not the whole project.

---

## Decision 3: The venv in Git Problem

**The situation:** Running `git status` revealed that `venv/` and `app/venv/` were being tracked — thousands of binary files that had no business being in version control.

**The investigation:** Before deleting anything, we checked which Python was actually being used. The active environment was `.venv/` (already gitignored). The committed `venv/` was Python 3.8 (dead), and `app/venv/` was empty. Both were safe to delete.

**The resolution:**
```bash
git rm -r --cached venv/ app/venv/
```

This removes them from git tracking while leaving the local files intact. Then `rm -rf` to clean them up entirely.

**The lesson:** A missing `.gitignore` is not just an aesthetic problem — it's a data integrity problem. Large generated directories in git history corrupt the repo's utility as a change-tracking tool. Create `.gitignore` before the first commit, not after.

---

## Decision 4: The Streamlit sys.path Bug

**The situation:** After moving the Streamlit app from `frontend.py` (at root) to `frontend/app.py`, it broke with:

```
ImportError: No module named 'api'
```

The same import that worked fine when running `uvicorn api.main:app` from the project root.

**The root cause:** When you run `streamlit run frontend/app.py`, Streamlit inserts `frontend/` into `sys.path` — not the project root. So Python can't find `api`, `data_pipeline`, or any other top-level package.

When you run `uvicorn api.main:app`, uvicorn doesn't manipulate `sys.path` — it expects you to be in the project root and uses the current directory as the base.

**The fix:**
```python
# frontend/app.py — top of file
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This resolves the path to the project root regardless of where the script is invoked from.

**The lesson:** Different runner tools have different `sys.path` conventions. When you move a script from one location to another, verify that all imports still resolve correctly from the new location. This is especially subtle in Python because `sys.path` is implicit.

---

## Decision 5: Pluggable LLM Backend

**The situation:** The app currently calls OpenRouter (cloud, costs money per call). The whole point of fine-tuning is to eventually serve the trained model locally via Ollama (free, private). This transition was coming.

**The decision:** Design the backend so switching from OpenRouter to Ollama requires changing exactly one line in `.env`, not touching any code.

```python
# api/config.py
class Settings(BaseSettings):
    llm_backend: str = "openrouter"  # or "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5"

# api/llm.py
async def generate_replies(user_message, conversation_id, ...):
    if settings.llm_backend == "ollama":
        from api.llm_local import generate_replies as ollama_generate
        return await ollama_generate(user_message, conversation_id)
    # ... OpenRouter logic
```

The Ollama client in `api/llm_local.py` has an identical function signature to the OpenRouter version. From the perspective of the API route, the backend is completely opaque.

**The trade-off:** A tiny bit of indirection now vs a code change and risk of regression later when switching models.

**The lesson:** If you know a component will change (the LLM backend will eventually be your fine-tuned model), design the seam upfront. The cost is a few extra lines. The benefit is that the swap becomes a configuration change, not a feature.

---

## Decision 6: PaddleOCR + KMeans over pytesseract + Hardcoded Color Thresholds

**The situation:** The original OCR pipeline used pytesseract with hardcoded HSV color range thresholds to distinguish which message bubble belongs to which sender. This broke on screenshots with different lighting, phone themes, or WeChat display settings.

**The new approach:** Two changes:
1. **PaddleOCR** instead of pytesseract — significantly better accuracy on Chinese text
2. **KMeans clustering** on HSV mean color instead of hardcoded thresholds

The KMeans insight: you don't need to know the absolute HSV values for "green" vs "white" bubbles. You just need to know that there are two clusters of bubble colors, and the higher-saturation cluster is the user's outgoing messages. This adapts automatically to any color scheme.

```python
color_data = np.array([[e["mean_h"], e["mean_s"], e["mean_v"]] for e in all_entries])
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(color_data)
user_cluster = int(np.argmax(kmeans.cluster_centers_[:, 1]))  # higher saturation = user
```

**New feature: OCR result caching.** OCR is the slow step (~1-5 seconds per image). Results are now saved to `_ocr_cache/<img>.json` alongside each image. Re-running the pipeline on already-processed images is instant. This means GPU is only needed once per image — subsequent runs (to adjust speaker labels, fix timestamps, etc.) are CPU-only.

**The trade-off:** More complex pipeline code, scikit-learn as a dependency. Worth it for robustness and re-run speed.

---

## Decision 7: Colab as GPU Runner, Not Dev Environment

**The situation:** Fine-tuning requires a GPU. Google Colab has free GPU. The developer's machine is a Mac. Claude Code (the AI assistant helping write code) runs in VSCode on the Mac.

The naive approach: edit code in Colab's notebook UI, run there. Problem: you lose IDE features, you lose Claude Code's ability to read and modify files, and every code change requires manual copy-paste between environments.

**The options considered:**

| Approach | Pros | Cons |
|----------|------|------|
| Edit in Colab | GPU available directly | No IDE, no Claude Code, manual copy-paste |
| Colab VSCode extension | IDE in VSCode | Only works for .ipynb notebooks, not .py scripts |
| Git-based workflow | Full IDE + Claude Code, version controlled | Requires a push + pull cycle per change |

**The decision:** Git-based workflow. Treat Colab as a GPU execution environment only, not a development environment.

```
VSCode (develop) → git push → Colab (execute)
  !git pull
  Run explore_v2.ipynb top-to-bottom
```

The key insight: the round-trip cost (push, switch to Colab, pull, run) is low compared to the cost of developing without an IDE. And because hyperparameters live in YAML config files — not in the script — most experiment iterations only require changing a YAML value and pushing, not touching Python code at all.

---

## Decision 8: Experiment Tracking Architecture

**The situation:** After the first training run, there was no way to answer: "What hyperparameters produced this model? What did the loss curve look like? Which system prompt was used?"

**The decision:** Three-layer tracking, all linked by a single `run_name`:

| What | Where | Tool |
|------|-------|------|
| Hyperparameters | `training/configs/qwen_v1.yaml` | Git |
| System prompt | `training/prompts/system_v1.txt` | Git |
| Loss curves + metrics | wandb.ai | WandB (free tier) |
| Model weights | HF Hub private repo | Hugging Face Hub |

Every experiment = one YAML file committed to git + one WandB run + one HF Hub model version, all sharing the `run_name` as identifier. To reconstruct exactly what produced any model: `git log training/configs/` → find the config → read it → find the WandB run by name → read the loss curve.

**New experiment workflow:**
```bash
cp training/configs/qwen_v1.yaml training/configs/qwen_v2.yaml
# Edit: increase LoRA rank from 16 to 32
git commit -m "experiment: qwen_v2 — higher LoRA rank"
git push
# In Colab: !git pull && run explore_v2.ipynb top-to-bottom
```

**The lesson:** In ML, reproducibility is correctness. A model that can't be reproduced is just an artifact. Treat experiment configs as code: version them, name them, link them to their outputs.

---

## Decision 9: Prompt v2 — From Rigid Tones to Emotional Intent

**The situation:** The inference prompt (what the live app uses to generate reply suggestions) had a fundamental structural problem: it offered three fixed tones — Warm, Playful, Formal. This framing sounds reasonable but produces robotic output. Real people don't choose a "tone" — they respond to what the other person needs.

The training prompt (`system_v1.txt`) had the right emotional foundation but only two examples, both light-hearted. No examples covering anxiety, excitement, or low-energy messages meant the fine-tuned model had a narrow behavioral range.

**The research:** Guided by OpenAI's GPT-4.1 prompting guide, several principles applied directly:

1. **Plan before act** — explicitly instruct the model to assess the sender's emotion and need before generating any reply
2. **Vary language deliberately** — adding an explicit instruction to vary vocabulary and structure prevents formulaic repetition
3. **Anchor at top and bottom** — the most critical instruction (sound like a real person) should appear twice: at role definition and again at the end
4. **Examples as executable specifications** — show the model what good looks like across different emotional registers, not just rules about what good is
5. **Specificity over length** — a clear sentence ("what do they need most right now?") outperforms a paragraph of guidance

**The decision:** Two separate rewrites, kept structurally consistent with each other.

*For the training prompt (`system_v2.txt`):*
- Added a formal 3-step **Emotional Assessment Protocol** that runs before every reply
- Expanded from 2 examples to 5, covering: light frustration, venting, anxiety/worry, good news, and ambiguous low-energy messages
- Bilingual: English instructions (clearer for LLM reasoning), Chinese examples (demonstrate the target texting style)
- Anchor at the bottom repeating the core identity

*For the inference prompt (`api/llm.py`):*
- Replaced Warm/Playful/Formal with three **intent-based labels**: "validates + opens up", "lifts the mood", "shows care simply"
- Added the same emotional assessment step before suggestions
- Added a full worked example showing the expected output shape
- Anchored with: "the goal is for the recipient to feel understood and want to keep the conversation going"

**The trade-off:** Intent labels are slightly less predictable than fixed tone labels from a UI perspective — the three suggestions won't always be in a neatly comparable format. But this is the right trade-off: coaching should be situational, not templated.

**Versioning:** `system_v1.txt` is preserved. A `qwen_v2.yaml` experiment config can point to `system_v2.txt` when ready to run a new training run on Colab.

**The lesson:** Prompts are specifications. The three-tone framework was a specification that happened to produce mediocre output. Replacing it with intent-based labels — and adding an explicit reasoning step before output — is the same structural insight as replacing a vague function signature with a well-typed one. The model follows instructions; the quality of the output is a function of the quality of those instructions.

---

---

## Decision 10: CPU Inference Timeout → Streaming Response

**The situation:** After switching the backend to Ollama with the fine-tuned GGUF, every request returned a generic `500 Internal Server Error`. There was no helpful error message in the frontend — just the raw FastAPI default.

**The diagnosis:** The server had no try/except around the LLM call, so all exceptions became opaque 500s. Adding a direct Python test reproduced the real error:

```
httpx.ReadTimeout
```

The `httpx.AsyncClient(timeout=60.0)` in `_call_ollama()` was expiring. The root cause: the machine running Ollama is an Intel i5-8257U (1.40 GHz, 2018 MacBook). Ollama confirmed this with `ollama ps`:

```
NAME                    PROCESSOR
qwen-emotional-coach    100% CPU
```

No GPU. No Metal. CPU-only inference at ~20 tokens/second. With no `num_predict` cap, the model generated 500+ tokens per response — 25+ seconds of pure generation, before even accounting for prompt evaluation. 60 seconds was never enough.

**The options:**
1. Raise the timeout to 120s → request still blocks; UX is a blank spinner for 30+ seconds
2. Add streaming → user sees tokens appear as they're generated; perceived wait is near-zero

**The decision:** Streaming. Two changes:

```python
# api/llm.py — new async generator
async def stream_ollama_reply(conversation_id: str):
    payload = {
        "model": settings.ollama_model,
        "messages": build_messages(conversation_id),
        "stream": True,
        "options": {"num_predict": 400},   # ← cap generation
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", ...) as resp:
            async for line in resp.aiter_lines():
                data = json.loads(line)
                if not data.get("done"):
                    yield data["message"]["content"]

# api/main.py — StreamingResponse for Ollama
@app.post("/suggest_reply")
async def suggest_reply(frd_msg: MessageRequest):
    if settings.llm_provider == "ollama":
        return StreamingResponse(
            stream_ollama_reply(frd_msg.conversation_id),
            media_type="text/plain",
        )
    result = await generate_replies(frd_msg.conversation_id)
    return {"reply": result.get("reply", "")}
```

Frontend uses `st.write_stream()` (available in Streamlit ≥ 1.31):

```python
st.write_stream(
    chunk for chunk in response.iter_content(chunk_size=None, decode_unicode=True)
    if chunk
)
```

**The lesson:** Streaming is not just a performance optimisation — it's a UX contract. On slow hardware, it's the difference between "this is broken" and "this is working, just a bit slow." Always check whether a blocking wait can become a stream. The code change is small; the perceived improvement is large.

---

## Decision 11: Conversation History Was Formatted Wrong for the Model

**The situation:** Even with streaming working, the model's replies were generic — not like the fine-tuned coaching style. The first instinct was that the GGUF export had failed to merge the LoRA weights. Checking the GGUF metadata and the notebook output confirmed the GGUF was fine.

**The real cause:** The training data was formatted as proper multi-turn ChatML conversations:

```
<|im_start|>system\n[prompt]<|im_end|>
<|im_start|>user\nfriend's message<|im_end|>
<|im_start|>assistant\nour reply<|im_end|>
<|im_start|>user\nnext friend message<|im_end|>
...
```

But `build_prompt()` collapsed all conversation history into the system message as plain text:

```python
# Old — broken
messages = [
    {"role": "system", "content": system_v2.txt + "\n\nRecent messages:\nfriend: X\nuser: Y"},
    {"role": "user",   "content": current_message},   # ← also duplicated in the system text above
]
```

The model was never trained to parse flat-text history inside a system prompt. It didn't recognise the context and fell back to generic base-model behaviour.

**The fix:** Split into two functions:

```python
def build_system_prompt(conversation_id: str) -> str:
    # Only the system instructions + optional prior summary
    ...

def build_messages(conversation_id: str) -> list[dict]:
    # Proper multi-turn array from DB history
    # DB sender "friend" → ChatML role "user"   (the person we reply to)
    # DB sender "user"   → ChatML role "assistant" (our previous replies)
    role_map = {"friend": "user", "user": "assistant"}
    ...
    return [
        {"role": "system", "content": build_system_prompt(conversation_id)},
        *chat_messages,
    ]
```

The current friend message is already the last DB entry (saved in the frontend before the API call), so no duplication is needed.

**The entity naming:** A subtlety worth spelling out for interviews:

| Entity | DB label | ChatML role | Reason |
|--------|----------|-------------|--------|
| Friend | `"friend"` | `"user"` | The person whose message we're responding to |
| App user | `"user"` | `"assistant"` | The LLM plays this persona — its output = the suggested reply |

This is counterintuitive (the "user" of the app maps to the "assistant" in the prompt) but follows directly from the training data, where the model was trained as the person replying, not the person asking.

**The lesson:** Inference format must exactly match training format. The model doesn't know what "Recent messages: friend: X" means — it only knows what it saw during training. If training used multi-turn ChatML, inference must too. This is easy to overlook because API-level abstractions (passing `messages` as a list) hide the fact that those messages are ultimately serialised into a specific token sequence — and the model is extremely sensitive to that sequence.

---

## Decision 12: The Missing Ollama Chat Template

**The situation:** After fixing the message format, responses were still wrong — they'd start with 1–2 sentences that looked natural, then continue with "As an AI assistant, I should mention..." paragraphs. The notebook's fine-tuned model produced clean, brief Chinese responses every time. What was different?

**The investigation:** `ollama show qwen-emotional-coach --modelfile` revealed the problem immediately:

```
TEMPLATE {{ .Prompt }}
```

This is Ollama's raw passthrough template. It means Ollama was not applying any ChatML formatting before passing the prompt to the model. The `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant` tokens — the exact delimiters the model was trained on — were never being injected.

Without those tokens, the model had no way to identify turn boundaries. It partially followed fine-tuned behaviour (the LoRA weights biased it toward the training distribution), but couldn't tell when a turn ended. Once the natural reply finished, the base Qwen RLHF training took over and produced generic assistant-style output. There was also no `<|im_end|>` stop token configured, so Ollama never halted generation cleanly.

The notebook worked because `tokenizer.apply_chat_template()` applied the correct formatting before inference. The Ollama path never had an equivalent step.

**Why it happened:** The `ollama/Modelfile` had no `TEMPLATE` directive. When `ollama create` was run, Ollama had no embedded template to fall back on (Unsloth's GGUF doesn't always carry a recognised template for custom-created models), so it defaulted to `{{ .Prompt }}` — a no-op that treats input as a raw pre-formatted string.

**The fix:** Add the Qwen2.5 ChatML template and stop token to the Modelfile, then recreate the model:

```
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
```

```bash
cd ollama && ollama create qwen-emotional-coach -f Modelfile
```

Verified with `ollama show` — the template was now applied correctly, and responses were brief, clean Chinese text matching the training style.

**The lesson:** When deploying a custom GGUF model in Ollama, always verify the template with `ollama show <model> --modelfile`. The default `{{ .Prompt }}` template silently produces wrong behaviour for any model that expects structured turn delimiters (ChatML, Llama, Gemma, etc.). The symptom — "good for a bit, then generic" — is the model being simultaneously pulled toward fine-tuned behaviour by LoRA weights and away from it by malformed input format. These two forces create exactly the observed pattern: partial success followed by base-model fallback.

---

---

## Session: 2026-04-04 — API Refactor & Prompt Architecture

### Decision 13: Validation Bug — Dead `message` Field Blocking LLM Calls

`MessageRequest.message` had `min_length=1` but the frontend was sending `""` — the friend's message was already saved to DB before `/suggest_reply` was called, so the field was never needed server-side. FastAPI was silently rejecting every request with 422 before the LLM was ever reached. Fixed by removing `min_length=1`, then later dropping the field entirely once confirmed dead.

---

### Decision 14: Friend Message Save — Direct DB Call Breaks in Production

**The problem:** `frontend/app.py` was calling `save_message()` from `api.db` directly — a Python function call that opens a SQLite file on the local filesystem. In production, Streamlit Cloud and Railway are separate servers with separate disks. The friend's message would write to Streamlit Cloud's ephemeral disk, invisible to the FastAPI backend on Railway.

**The decision:** Added `POST /friend_message` endpoint. Frontend now makes an HTTP request like every other action. All DB writes flow through Railway's disk. The frontend no longer imports anything from `api.db`.

---

### Decision 15: Merged `/log_suggestion` into `/suggest_reply`

**The problem:** After streaming, the frontend reassembled the full reply text client-side and sent it back to the backend via a second HTTP call just to save it. The backend had generated that text and discarded it.

**The decision:** Backend accumulates chunks during streaming, saves to DB itself, and emits a final NDJSON line `{"done": true, "suggestion_id": N}`. Non-streaming path includes `suggestion_id` directly in the JSON response. One round trip instead of two. `LogSuggestionRequest` and the `/log_suggestion` endpoint deleted entirely.

**Streaming format change:** Plain text chunks → NDJSON (`{"chunk": "..."}` lines). Gives the frontend a structured protocol to display chunks and extract the suggestion ID cleanly from the terminal line.

---

### Decision 16: Kept `sent=0` Rows as Training Data

Initially flagged as DB bloat. Reversed after reflection — `sent=0` unambiguously means "LLM suggested this, user chose not to use it." This is a valuable negative training signal. No cleanup logic added. Future training queries:

```sql
SELECT * FROM messages WHERE sender='llm' AND sent=1;  -- accepted (positive)
SELECT * FROM messages WHERE sender='llm' AND sent=0;  -- rejected (negative)
```

---

### Decision 17: Chat UI Placeholder Moved to Top of Script

`display_chat_ui()` was called at the bottom of the Streamlit script. During streaming the script was blocked mid-execution — the placeholder stayed empty until streaming finished, making the chat window disappear. Moved the call immediately after `st.empty()` so chat history renders before streaming begins and stays visible throughout.

---

### Decision 18: Input Box Wrapped in `st.form`

Streamlit text areas don't commit their value to session state until the widget loses focus (blur). Users had to click outside the box before clicking a button — an unintuitive UX gap. Wrapping in `st.form` batches the text area value and button click together so clicking submit commits both simultaneously without needing to click away.

---

### Decision 19: Conversation Summarizer

**The problem:** `get_last_messages(n=20)` creates a hard cutoff — everything before message 20 is invisible to the LLM. Longer conversations lose context.

**The decision:** `maybe_summarize()` runs before every LLM call. Trigger: `(total_messages - 20) % 5 == 0` — once every 5 new messages past the window. Incremental: passes existing summary + new overflow to the LLM so each new summary builds on the last rather than reprocessing the entire history. Saved to the existing `summaries` table in SQLite.

---

### Decision 20: History Embedded in System Prompt, Not as Separate ChatML Turns

**Context:** Decision 11 established that history should be proper ChatML turns, not flat text in the system message. This was the right call at the time — training data used multi-turn ChatML. With the introduction of `{HISTORY}` as a system prompt placeholder, we reversed this.

**The reasoning:** The `{HISTORY}` placeholder formats messages as `Friend: ...` / `You: ...` lines directly inside the system message. This removes the separate `user`/`assistant` turn array from the payload entirely. Benefits: the debug terminal log is fully self-contained (one JSON block shows everything), no duplication, and the prompt caching architecture (static content first, dynamic content last) is easier to maintain in one place. The trade-off is a departure from strict ChatML formatting — acceptable because the production model (vLLM/RunPod) will be a general instruction model handling this naturally, and the Ollama path is already on custom GGUF with a configured template.

---

### Decision 21: Standardised Naming in system_v3.txt

**The problem:** Examples used `U:` / `A:` (ChatML jargon — "User", "Assistant"), while the `{HISTORY}` block produced by the code used `Friend:` / `You:`. The LLM saw different naming conventions in the instructions versus the actual conversation data it was replying to — a subtle inconsistency that could cause the model to treat the examples as a different format than the real context.

**The decision:** Replaced all `U:` → `Friend:`, all `A:` → `You:` throughout the prompt. Defined both terms explicitly at the top as a glossary. Every reference in instructions, examples, and section headers now uses the same two labels the LLM sees in `{HISTORY}`.

---

## What's Next

**Completed:**
- ✅ First training run — 73 conversations, 500 steps, LoRA r=16 on attention layers
- ✅ GGUF export and Ollama integration
- ✅ Streaming implementation (CPU inference workaround)
- ✅ Multi-turn message format fixed (build_messages with proper role mapping)
- ✅ ChatML template added to Modelfile (biggest quality fix)

**Remaining:**
- Evaluate current model quality in context — does it sound like the target communication style?
- Iterate on training: more data, higher LoRA rank, MLP layers included, system_v2.txt in training data
- For next training run: embed `system_v2.txt` in the training data's system messages (currently training data uses the original Chinese system prompt, creating a training/inference mismatch)
- Production: push merged model to HF Hub → vLLM on RunPod → switch `LLM_PROVIDER=vllm` in Railway `.env`
