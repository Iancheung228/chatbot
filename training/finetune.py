"""
Fine-tuning script for Qwen with Unsloth + LoRA.

Workflow:
  1. Edit code/config/prompt in VSCode (with Claude Code's help)
  2. git push
  3. In Google Colab (GPU runtime):
       !git pull
       !pip install -r requirements-training.txt -q
       !python training/finetune.py --config training/configs/qwen_v1.yaml
  4. Weights auto-pushed to HF Hub (if enabled in config)
  5. Training curves visible at wandb.ai (if enabled in config)

New experiment = copy qwen_v1.yaml → qwen_v2.yaml, edit one line, commit.
"""
import argparse
import os
import sys

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_output_dir(cfg: dict) -> str:
    return cfg["output"]["dir"].format(run_name=cfg["run_name"])


def print_config(cfg: dict, output_dir: str) -> None:
    print("=" * 60)
    print(f"  run_name   : {cfg['run_name']}")
    print(f"  model      : {cfg['model']['name']}")
    print(f"  data       : {cfg['data']['input_jsonl']}")
    print(f"  prompt     : {cfg['prompt']['system_file']}")
    print(f"  output_dir : {output_dir}")
    print(f"  max_steps  : {cfg['training']['max_steps']}")
    print(f"  lr         : {cfg['training']['learning_rate']}")
    print(f"  lora_r     : {cfg['lora']['r']}")
    print(f"  wandb      : {'enabled' if cfg['wandb']['enabled'] else 'disabled'}")
    print(f"  push_hub   : {'enabled' if cfg['output']['push_to_hub'] else 'disabled'}")
    print("=" * 60)


def setup_wandb(cfg: dict) -> None:
    if cfg["wandb"]["enabled"]:
        import wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["run_name"],
            config={
                "model": cfg["model"],
                "lora": cfg["lora"],
                "training": cfg["training"],
                "data": cfg["data"],
            },
        )
        os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
    else:
        os.environ["WANDB_DISABLED"] = "true"


def train(cfg: dict) -> None:
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ---------- load system prompt ----------
    with open(cfg["prompt"]["system_file"], "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # ---------- load dataset ----------
    jsonl_path = cfg["data"]["input_jsonl"]
    if not os.path.exists(jsonl_path):
        print(f"ERROR: training data not found at '{jsonl_path}'")
        print("Run the data pipeline first:")
        print("  python -m data_pipeline.process --root_folder <path> --out_jsonl output/unsloth_chatml.jsonl")
        sys.exit(1)

    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    print(f"Loaded {len(dataset)} conversations from {jsonl_path}")

    # ---------- load model ----------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        device_map="auto",
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )

    # ---------- inject system prompt + format ----------
    def format_chat(example):
        messages = example["messages"]
        # replace or inject system prompt from file
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages = [{"role": "system", "content": system_prompt}] + messages

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat, batched=False)

    # ---------- apply LoRA ----------
    lora_cfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
    )

    # ---------- training ----------
    output_dir = resolve_output_dir(cfg)
    t_cfg = cfg["training"]
    training_args = TrainingArguments(
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        warmup_steps=t_cfg["warmup_steps"],
        max_steps=t_cfg["max_steps"],
        learning_rate=t_cfg["learning_rate"],
        fp16=t_cfg["fp16"],
        logging_steps=t_cfg["logging_steps"],
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=t_cfg["save_steps"],
        report_to="wandb" if cfg["wandb"]["enabled"] else "none",
        run_name=cfg["run_name"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        args=training_args,
        train_on_completions_only=True,
    )

    trainer.train()

    # ---------- save final adapter ----------
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved LoRA adapter to: {final_path}")

    # ---------- push to HF Hub ----------
    if cfg["output"]["push_to_hub"]:
        hub_repo = cfg["output"]["hub_repo"]
        private = cfg["output"].get("hub_private", True)
        hf_token = os.environ.get("HF_TOKEN")
        print(f"Pushing to HF Hub: {hub_repo} ...")
        model.push_to_hub(hub_repo, token=hf_token, private=private)
        tokenizer.push_to_hub(hub_repo, token=hf_token, private=private)
        print(f"Done. Model available at: https://huggingface.co/{hub_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with Unsloth.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without training",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = resolve_output_dir(cfg)
    print_config(cfg, output_dir)

    if args.dry_run:
        print("Dry run — exiting without training.")
        return

    setup_wandb(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
