"""
MistralMind - Specialist Fine-Tuning Script
============================================
Fine-tunes mistralai/Mistral-7B-Instruct-v0.3 into domain specialists
using Unsloth + TRL. Each specialist tracked independently with W&B.

Specialists & Datasets:
  🏥 medical   : qiaojin/PubMedQA
  📈 finance   : gbharti/finance-alpaca
  💻 code      : sahil2801/CodeAlpaca-20k
  🎨 creative  : Dahoas/synthetic-instruct-gptj-pairwise

Fine-Tuning Setup (as per spec):
  Base model : mistralai/Mistral-7B-Instruct-v0.3   ✅
  Method     : QLoRA (4-bit) + Unsloth               ✅
  LoRA rank  : r=64, alpha=128, RSLoRA enabled        ✅
  Packing    : Sequence packing for GPU efficiency    ✅
  Tracking   : W&B with custom eval dashboards        ✅

Usage:
  python finetune/train_specialist.py --specialist medical  --epochs 3
  python finetune/train_specialist.py --specialist finance  --epochs 3
  python finetune/train_specialist.py --specialist code     --epochs 3
  python finetune/train_specialist.py --specialist creative --epochs 3
"""

import os
import argparse
import torch
import wandb
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Global Config  ← CORRECTED: Mistral-7B, r=64, alpha=128, seq=4096
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"  # ← FIXED (was Llama)
MAX_SEQ_LEN  = 4096                                   # ← FIXED (was 2048)
LOAD_IN_4BIT = True
LORA_RANK    = 64                                     # ← FIXED (was 32)
LORA_ALPHA   = 128                                    # ← FIXED (was 64)
OUTPUT_DIR   = "./checkpoints"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# ─────────────────────────────────────────────────────────────────────────────
# Specialist Configurations
# ─────────────────────────────────────────────────────────────────────────────

SPECIALIST_CONFIG = {
    "medical": {
        "dataset_id":   "qiaojin/PubMedQA",
        "dataset_name": "pqa_labeled",
        "split":        "train",
        "system_prompt": (
            "You are a world-class medical expert and clinical reasoning specialist. "
            "You think step by step using evidence-based medicine. You cite relevant studies, "
            "explain pathophysiology clearly, and always prioritize patient safety. "
            "State your confidence level and flag when specialist consultation is needed."
        ),
        "learning_rate": 2e-4,
    },
    "finance": {
        "dataset_id":   "gbharti/finance-alpaca",
        "dataset_name": None,
        "split":        "train",
        "system_prompt": (
            "You are a senior quantitative analyst and financial strategist with deep expertise "
            "in markets, valuation, risk management, and macroeconomics. "
            "Provide rigorous, data-driven analysis with clear reasoning chains. "
            "Always quantify uncertainty and explicitly flag key assumptions."
        ),
        "learning_rate": 2e-4,
    },
    "code": {
        "dataset_id":   "sahil2801/CodeAlpaca-20k",
        "dataset_name": None,
        "split":        "train",
        "system_prompt": (
            "You are an expert software engineer with mastery across languages and paradigms. "
            "Write clean, efficient, well-documented code. Explain your reasoning thoroughly, "
            "identify edge cases proactively, and follow industry best practices. "
            "Think like a senior engineer conducting a careful code review."
        ),
        "learning_rate": 2e-4,
    },
    "creative": {
        "dataset_id":   "Dahoas/synthetic-instruct-gptj-pairwise",
        "dataset_name": None,
        "split":        "train",
        "system_prompt": (
            "You are a brilliant creative writer with a distinctive voice and vivid imagination. "
            "Craft engaging narratives, evocative descriptions, and compelling characters. "
            "Use literary techniques masterfully — show don't tell, subtext, rhythm, metaphor. "
            "Adapt your style fluidly to match any genre, tone, or creative constraint."
        ),
        "learning_rate": 1e-4,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Formatters
# ─────────────────────────────────────────────────────────────────────────────

def format_pubmedqa(example):
    """PubMedQA: pubid, question, context (dict), long_answer, final_decision"""
    question     = example.get("question", "")
    answer       = example.get("long_answer", "")
    context_dict = example.get("context", {})
    contexts     = context_dict.get("contexts", []) if isinstance(context_dict, dict) else []
    context_str  = " ".join(contexts[:3])

    if not answer or len(answer) < 20:
        return None

    instruction = (
        f"Research Context:\n{context_str}\n\nClinical Question:\n{question}"
        if context_str else question
    )
    return {"instruction": instruction, "response": answer}


def format_finance_alpaca(example):
    """finance-alpaca: instruction, input, output"""
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()
    if inp:
        instruction = f"{instruction}\n\nContext: {inp}"
    if not instruction or not output or len(output) < 10:
        return None
    return {"instruction": instruction, "response": output}


def format_code_alpaca(example):
    """CodeAlpaca-20k: instruction, input, output"""
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()
    if inp:
        instruction = f"{instruction}\n\n```\n{inp}\n```"
    if not instruction or not output or len(output) < 10:
        return None
    return {"instruction": instruction, "response": output}


def format_creative_pairwise(example):
    """synthetic-instruct-gptj-pairwise: prompt, chosen, rejected"""
    prompt = example.get("prompt", "").strip()
    chosen = example.get("chosen", "").strip()
    if not prompt or not chosen or len(chosen) < 20:
        return None
    prompt = prompt.replace("Human:", "").replace("Assistant:", "").strip()
    return {"instruction": prompt, "response": chosen}


FORMATTERS = {
    "medical":  format_pubmedqa,
    "finance":  format_finance_alpaca,
    "code":     format_code_alpaca,
    "creative": format_creative_pairwise,
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_specialist_dataset(specialist: str, max_samples: int = 10_000) -> Dataset:
    cfg       = SPECIALIST_CONFIG[specialist]
    formatter = FORMATTERS[specialist]

    print(f"  📥 Loading {cfg['dataset_id']} ...")
    load_kwargs = {
        "path":              cfg["dataset_id"],
        "split":             cfg["split"],
        "trust_remote_code": True,
    }
    if cfg["dataset_name"]:
        load_kwargs["name"] = cfg["dataset_name"]

    ds        = load_dataset(**load_kwargs)
    formatted = [r for row in ds if (r := formatter(row)) is not None]
    ds_clean  = Dataset.from_list(formatted).shuffle(seed=42)

    if len(ds_clean) > max_samples:
        ds_clean = ds_clean.select(range(max_samples))

    print(f"  ✅ {specialist}: {len(ds_clean):,} samples ready")
    return ds_clean

# ─────────────────────────────────────────────────────────────────────────────
# Chat Formatter  ← uses Mistral template
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_formatter(tokenizer, system_prompt: str):
    def format_for_sft(example):
        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}
    return format_for_sft

# ─────────────────────────────────────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_specialist(specialist: str, epochs: int = 3, max_samples: int = 10_000):
    cfg = SPECIALIST_CONFIG[specialist]

    # W&B
    wandb.init(
        project = os.getenv("WANDB_PROJECT", "mistralmind"),
        name    = f"{specialist}-specialist-v1",
        tags    = ["mistralmind", specialist, "unsloth", "qlora", "mistral-7b"],
        config  = {
            "specialist":    specialist,
            "dataset":       cfg["dataset_id"],
            "base_model":    BASE_MODEL,
            "epochs":        epochs,
            "max_seq_len":   MAX_SEQ_LEN,
            "lora_rank":     LORA_RANK,
            "lora_alpha":    LORA_ALPHA,
            "learning_rate": cfg["learning_rate"],
            "max_samples":   max_samples,
            "load_in_4bit":  LOAD_IN_4BIT,
            "rslora":        True,
            "packing":       True,
        }
    )

    # Load model with Unsloth
    print(f"\n🔧 Loading {BASE_MODEL} for [{specialist}]...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,
        load_in_4bit   = LOAD_IN_4BIT,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")  # ← FIXED

    # QLoRA with RSLoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_RANK,
        target_modules             = TARGET_MODULES,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = 0.05,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
        use_rslora                 = True,   # ← Rank-Stabilized LoRA
    )

    # Dataset
    print(f"\n📚 Preparing [{specialist}] dataset...")
    dataset   = load_specialist_dataset(specialist, max_samples)
    formatter = build_chat_formatter(tokenizer, cfg["system_prompt"])
    dataset   = dataset.map(formatter, remove_columns=dataset.column_names)
    split     = dataset.train_test_split(test_size=0.05, seed=42)
    train_d, eval_d = split["train"], split["test"]
    print(f"   Train: {len(train_d):,} | Eval: {len(eval_d):,}")

    # Log sample rows to W&B
    wandb.log({"training_samples": wandb.Table(
        columns=["text"],
        data=[[train_d[i]["text"]] for i in range(min(5, len(train_d)))]
    )})

    # Training args
    output_path   = os.path.join(OUTPUT_DIR, specialist)
    training_args = TrainingArguments(
        output_dir                  = output_path,
        num_train_epochs            = epochs,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size  = 2,
        gradient_accumulation_steps = 8,
        warmup_ratio                = 0.05,
        learning_rate               = cfg["learning_rate"],
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        eval_strategy               = "steps",
        eval_steps                  = 100,
        save_strategy               = "steps",
        save_steps                  = 200,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "wandb",
        optim                       = "adamw_8bit",
        lr_scheduler_type           = "cosine",
        seed                        = 42,
        dataloader_num_workers      = 4,
        group_by_length             = True,
    )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_d,
        eval_dataset       = eval_d,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LEN,
        args               = training_args,
        packing            = True,   # ← sequence packing for efficiency
    )

    print(f"\n🚀 Training [{specialist}] — {epochs} epochs...")
    stats = trainer.train()

    wandb.log({
        "final_train_loss":      stats.training_loss,
        "total_steps":           stats.global_step,
        "train_runtime_seconds": stats.metrics["train_runtime"],
        "samples_per_second":    stats.metrics["train_samples_per_second"],
    })

    # Save LoRA adapter
    lora_path   = os.path.join(output_path, "lora_adapter")
    merged_path = os.path.join(output_path, "merged_16bit")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"   💾 LoRA   → {lora_path}")
    print(f"   💾 Merged → {merged_path}")

    # Optional HuggingFace push
    hf_token, hf_user = os.getenv("HF_TOKEN"), os.getenv("HF_USERNAME")
    if hf_token and hf_user:
        repo_id = f"{hf_user}/mistralmind-{specialist}"
        model.push_to_hub_merged(repo_id, tokenizer, save_method="lora", token=hf_token)
        print(f"   📤 HF → {repo_id}")
        wandb.log({"hf_model_url": f"https://huggingface.co/{repo_id}"})

    wandb.finish()
    print(f"\n✅ [{specialist}] complete!")
    return lora_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", choices=list(SPECIALIST_CONFIG.keys()), required=True)
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=10_000)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MistralMind — [{args.specialist.upper()}] Specialist")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Dataset    : {SPECIALIST_CONFIG[args.specialist]['dataset_id']}")
    print(f"  LoRA       : r={LORA_RANK}, α={LORA_ALPHA}, RSLoRA=True")
    print(f"  Epochs     : {args.epochs} | Samples: {args.max_samples:,}")
    print(f"{'='*60}")
    train_specialist(args.specialist, args.epochs, args.max_samples)
