#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_sft.py — 监督微调 (SFT) + LoRA
修复点：
- LoRA 配置: r=16, alpha=32, lr=2e-4（匹配项目设定）
- 早停回调正确注册到 Trainer
- 修复 state 变量未定义的 NameError
- 使用 DataCollator 动态 padding（不再每条填满 2048）
- 计算并记录验证集 PPL
- 模型名修正为 Qwen1.5-1.8B
"""
import json
import os
import math
import time
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore")

ROOT = "D:/SFT"
TRAIN_PATH = os.path.join(ROOT, "data/processed/train.jsonl")
VAL_PATH = os.path.join(ROOT, "data/processed/val.jsonl")
TEST_PATH = os.path.join(ROOT, "data/processed/test.jsonl")
MODEL_NAME = "Qwen/Qwen1.5-1.8B"
MAX_LEN = 2048

# 超参数（匹配项目设定）
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4

PER_DEVICE_BS = 4
GRAD_ACC = 8  # effective bs = 32

OUT_DIR_BASE = os.path.join(ROOT, "models/sft")
TENSORBOARD_DIR = os.path.join(ROOT, "logs/sft")
os.makedirs(OUT_DIR_BASE, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)


# ==================== Dataset ====================
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.examples.append(obj)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        question = item["question"]
        answer = item["answer"]

        # ChatML 格式
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        full = f"{prompt}{answer}<|im_end|>\n"

        prompt_ids = self.tokenizer(
            prompt, truncation=True, max_length=self.max_len,
            return_tensors="pt", add_special_tokens=False,
        )["input_ids"].squeeze()

        full_ids = self.tokenizer(
            full, truncation=True, max_length=self.max_len,
            return_tensors="pt", add_special_tokens=False,
        )["input_ids"].squeeze()

        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100  # 只计算 answer 部分的 loss

        return {
            "input_ids": full_ids,
            "labels": labels,
        }


# ==================== 早停 + PPL 回调 ====================
class EarlyStopPPLCallback(TrainerCallback):
    """监控验证 PPL，连续 3 轮不降则早停"""

    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_ppl = float("inf")
        self.best_loss = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_loss = metrics.get("eval_loss", float("inf"))
        eval_ppl = math.exp(min(eval_loss, 20))  # 避免溢出

        print(f"\n  [Eval] loss={eval_loss:.4f}, ppl={eval_ppl:.2f}, "
              f"best_ppl={self.best_ppl:.2f}, counter={self.counter}/{self.patience}")

        if eval_ppl < self.best_ppl - 0.01:
            self.best_ppl = eval_ppl
            self.best_loss = eval_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"  [EarlyStop] PPL 连续 {self.patience} 轮未改善，停止训练")
            control.should_training_stop = True

        return control


# ==================== 训练 ====================
def train():
    print("=" * 60)
    print("STAGE 2: SFT + LoRA Training")
    print(f"  Model: {MODEL_NAME}")
    print(f"  LoRA: r={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  LR: {LEARNING_RATE}, batch_size={PER_DEVICE_BS}x{GRAD_ACC}")
    print("=" * 60)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    print("[SFT] Loading datasets...")
    train_dataset = SFTDataset(TRAIN_PATH, tokenizer)
    val_dataset = SFTDataset(VAL_PATH, tokenizer)
    print(f"[SFT] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("[SFT] ERROR: No training data!")
        return

    # Model
    print("[SFT] Loading model...")
    # 显式通过 AutoConfig 设置 attn_implementation，避免 transformers 5.x 默认配置覆盖
    attn_impl = "sdpa" if torch.cuda.is_available() else "eager"
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config._attn_implementation = attn_impl
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    run_dir = os.path.join(OUT_DIR_BASE, "run_main")
    os.makedirs(run_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=5,
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRAD_ACC,
        optim="adamw_torch",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        bf16=False,
        disable_tqdm=False,
        seed=42,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_LEN,
    )

    callback = EarlyStopPPLCallback(patience=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[callback],  # 关键：注册回调
    )

    # Train
    print("[SFT] Start training...")
    result = trainer.train()

    # 保存最佳模型
    best_dir = os.path.join(OUT_DIR_BASE, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # 计算最终 PPL
    print("\n[SFT] Computing final PPL on val set...")
    eval_results = trainer.evaluate()
    final_loss = eval_results.get("eval_loss", float("inf"))
    final_ppl = math.exp(min(final_loss, 20))

    # 保存训练摘要
    summary = {
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": 5,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_runtime": result.metrics.get("train_runtime", 0),
        "final_train_loss": result.metrics.get("train_loss", 0),
        "final_eval_loss": final_loss,
        "final_eval_ppl": final_ppl,
        "best_eval_ppl": callback.best_ppl,
    }

    summary_path = os.path.join(OUT_DIR_BASE, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"[SFT] Training Complete!")
    print(f"  Final eval loss: {final_loss:.4f}")
    print(f"  Final eval PPL:  {final_ppl:.2f}")
    print(f"  Best eval PPL:   {callback.best_ppl:.2f}")
    print(f"  Best model saved -> {best_dir}")
    print(f"  Summary -> {summary_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    train()
