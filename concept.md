# Concepts

ML concepts and gotchas encountered while building this project.

---

## LoRA: when is the adapter applied?

After `FastLanguageModel.get_peft_model()`, the LoRA adapter is already attached — `model` is the base model + adapter combined. The base weights are frozen; only the adapter parameters are trainable.

`model.print_trainable_parameters()` will show something like:

```
trainable params: 2,621,440 || all params: 1,546,129,408 || trainable%: 0.17%
```

That ~0.17% is the LoRA adapter. The other 99.83% (base weights) are frozen and won't be updated during training.

**Key nuance:** even though the adapter is attached, the model behaves identically to the base model at this point because LoRA initialises matrix B to zeros, making the adapter output `ΔW = B·A = 0`. It only diverges from the base as training updates B and A away from their initial values.

| After call | State |
|---|---|
| `FastLanguageModel.from_pretrained()` | Pure base model, all weights trainable |
| `FastLanguageModel.get_peft_model()` | Base (frozen) + LoRA adapter (trainable), but output = base model output |
| `trainer.train()` | Adapter weights diverge — model now differs from base |

---

## `save_pretrained` vs `push_to_hub` vs `push_to_hub_merged` — what gets saved?

These three do very different things depending on what `model` is at call time.

**`model.save_pretrained(path)`** — saves to local disk.
If `model` is a `PeftModel`, saves only the LoRA adapter weights (~10–50 MB), not the base model:
```
adapter_config.json          ← records r, lora_alpha, target_modules, base model name
adapter_model.safetensors    ← just the A and B matrices
```

**`model.push_to_hub(repo)`** — same as above but uploads to HuggingFace Hub.
Only the adapter lands on HF. To run it, you must load the base model and attach the adapter separately:
```python
model = AutoModelForCausalLM.from_pretrained("Qwen2.5-1.5B")   # base
model = PeftModel.from_pretrained(model, "yourname/adapter-repo")  # attach
```

**`model.push_to_hub_merged(repo)`** — Unsloth-specific. Merges adapter into base first, then pushes the full model (~3 GB for Qwen 1.5B in 16-bit). Self-contained — no adapter or base model needed separately:
```python
model = AutoModelForCausalLM.from_pretrained("yourname/merged-repo")  # just works
```

| Method | Destination | What's saved | Size | Needs base model to run? |
|---|---|---|---|---|
| `save_pretrained()` | Local disk | Adapter only | ~10–50 MB | Yes |
| `push_to_hub()` | HF Hub | Adapter only | ~10–50 MB | Yes |
| `push_to_hub_merged()` | HF Hub | Full merged model | ~3 GB | No |

In the notebook all three are used: `save_pretrained` as a local backup, `push_to_hub` to version the adapter, `push_to_hub_merged` to create the production model.

---

## How `push_to_hub_merged` knows to merge base + adapter

It knows because of what `model` **is** as a Python object at that point.

After `get_peft_model()`, `model` is no longer a plain `PreTrainedModel` — it's a `PeftModel` instance that wraps the base model. Internally it holds two things simultaneously:

- the frozen base weights (`model.base_model`)
- the trained LoRA matrices A and B for each targeted layer

When you call `push_to_hub_merged`, Unsloth inspects the object type, sees it's a `PeftModel`, and runs the merge before pushing:

```
W_merged = W_base + (B · A) * (lora_alpha / r)
```

for every targeted layer (`q_proj`, `k_proj`, etc.). The result is a single weight matrix per layer with the fine-tuning baked in — no separate adapter needed at inference time.

If you passed a plain base model that had never gone through `get_peft_model()`, calling `push_to_hub_merged` would just push the base weights unchanged (nothing to merge).
