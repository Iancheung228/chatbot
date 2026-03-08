"""
Fine-tuning script for Qwen with Unsloth.

Workflow:
  1. Edit this script in VSCode
  2. Push to GitHub
  3. In Google Colab: !git pull && python training/finetune.py --config training/configs/qwen_config.yaml
  4. Weights are saved to Google Drive (path configured in qwen_config.yaml)

This script is a stub — populate it with your Colab notebook code.
"""
import argparse
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(config: dict):
    # TODO: paste your Unsloth fine-tuning code here
    # from unsloth import FastLanguageModel
    # model, tokenizer = FastLanguageModel.from_pretrained(...)
    # ...
    print("Training config:", config)
    raise NotImplementedError("Populate this with your Unsloth training code from Colab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
