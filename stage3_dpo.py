#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage3_dpo.py — 偏好对齐（DPO）
替代 PPO 的更稳定方案：
1. 从 SFT 数据构造偏好对（chosen = 原始高质量回答，rejected = 退化回答）
2. 使用 TRL 的 DPOTrainer 训练
3. 训练后模型回答更符合用户意图
"""
import json
import os
import random
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

warnings.filterwarnings("ignore")

ROOT = "D:/SFT"
TRAIN_PATH = os.path.join(ROOT, "data/processed/train.jsonl")
VAL_PATH = os.path.join(ROOT, "data/processed/val.jsonl")
SFT_MODEL_PATH = os.path.join(ROOT, "models/sft/best")
OUT_DIR = os.path.join(ROOT, "models/dpo")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "Qwen/Qwen1.5-1.8B"


def build_preference_pairs(data_path, max_pairs=500):
    """
    从 SFT 数据构造偏好对。
    chosen = 原始回答
    rejected = 退化回答（截断、去结构化、泛化回答）
    """
    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    pairs = []
    for item in items[:max_pairs]:
        q = item["question"]
        a = item["answer"]

        # 构造 rejected：截断 + 去代码块标记，模拟低质量回答
        rejected = a[:len(a) // 2]  # 截断一半
        rejected = rejected.replace("```", "")  # 去代码块
        rejected = rejected.replace("1.", "").replace("2.", "").replace("3.", "")  # 去序号
        rejected = rejected.strip()
        if len(rejected) < 10:
            rejected = "这个问题的答案比较复杂，需要具体情况具体分析。"

        # ChatML prompt（与 stage2 SFT 训练格式一致：无 system）
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

        pairs.append({
            "prompt": prompt,
            "chosen": a,
            "rejected": rejected,
        })

    random.shuffle(pairs)
    return pairs


# ==================== 手写 DPO（不依赖 TRL） ====================
MAX_LEN = 768
MAX_PROMPT_LEN = 384


class DPODataset(TorchDataset):
    """把 prompt/chosen/rejected 转成 token id（labels 只对回答部分计分）"""

    def __init__(self, pairs, tokenizer):
        self.examples = []
        for p in pairs:
            prompt_ids = tokenizer(
                p["prompt"], add_special_tokens=False,
                truncation=True, max_length=MAX_PROMPT_LEN,
            )["input_ids"]
            win_ids = tokenizer(
                p["chosen"] + "<|im_end|>", add_special_tokens=False,
                truncation=True, max_length=MAX_LEN - MAX_PROMPT_LEN,
            )["input_ids"]
            lose_ids = tokenizer(
                p["rejected"] + "<|im_end|>", add_special_tokens=False,
                truncation=True, max_length=MAX_LEN - MAX_PROMPT_LEN,
            )["input_ids"]
            self.examples.append((prompt_ids, win_ids, lose_ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def dpo_collator(batch, pad_id):
    """分别对 chosen / rejected 序列做 left-free padding，prompt 拼在前面"""
    def build(key_idx):
        seqs = [list(p[0]) + list(p[key_idx]) for p in batch]
        maxlen = max(len(s) for s in seqs)
        input_ids, labels, attn = [], [], []
        for s, p in zip(seqs, batch):
            plen = len(p[0])
            pad_n = maxlen - len(s)
            input_ids.append(s + [pad_id] * pad_n)
            lab = [-100] * plen + s[plen:] + [-100] * pad_n
            labels.append(lab)
            attn.append([1] * len(s) + [0] * pad_n)
        return (torch.tensor(input_ids), torch.tensor(labels),
                torch.tensor(attn))

    wi, wl, wm = build(1)
    li, ll, lm = build(2)
    return {
        "win_input_ids": wi, "win_labels": wl, "win_mask": wm,
        "lose_input_ids": li, "lose_labels": ll, "lose_mask": lm,
    }


def sequence_logps(model, input_ids, attn_mask, labels):
    """计算 completion 部分的 token log 概率之和"""
    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[:, :-1, :]
    labels = labels[:, 1:]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    per_tok = logprobs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
    return (per_tok * mask).sum(dim=-1)


class DPOTrainer(Trainer):
    """DPO 损失：-log sigmoid(beta * [(π_w-πref_w) - (π_l-πref_l)])"""

    def __init__(self, *args, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        w = (inputs["win_input_ids"], inputs["win_mask"], inputs["win_labels"])
        l = (inputs["lose_input_ids"], inputs["lose_mask"], inputs["lose_labels"])

        # reference logps：禁用 LoRA adapter 即得到冻结的 SFT 模型（无梯度）
        with torch.no_grad(), model.disable_adapter():
            ref_w = sequence_logps(model, *w)
            ref_l = sequence_logps(model, *l)

        # policy logps：启用 adapter（有梯度）
        pol_w = sequence_logps(model, *w)
        pol_l = sequence_logps(model, *l)

        logits = self.beta * ((pol_w - ref_w) - (pol_l - ref_l))
        loss = -F.logsigmoid(logits).mean()
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        """评估时输入是自定义字段，不能走默认 prediction_step，直接复用 DPO 损失"""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs).mean().detach()
        return (loss, None, None)


def train_dpo():
    print("=" * 60)
    print("STAGE 3: DPO Preference Alignment (hand-rolled, no TRL)")
    print("=" * 60)

    # 构造偏好数据
    print("[DPO] Building preference pairs...")
    train_pairs = build_preference_pairs(TRAIN_PATH, max_pairs=400)
    val_pairs = build_preference_pairs(VAL_PATH, max_pairs=50)
    print(f"[DPO] Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = DPODataset(train_pairs, tokenizer)
    val_ds = DPODataset(val_pairs, tokenizer)

    # 1) 加载基座 + 合并 SFT adapter，得到权重=SFT 的普通模型
    print("[DPO] Loading base model and merging SFT adapter...")
    attn_impl = "sdpa" if torch.cuda.is_available() else "eager"
    base_config = AutoConfig.from_pretrained(
        MODEL_NAME, trust_remote_code=True, local_files_only=True
    )
    base_config._attn_implementation = attn_impl
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=base_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        local_files_only=True,
    )
    sft_model = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH)
    merged = sft_model.merge_and_unload()  # 权重已合并为 SFT
    del sft_model

    # 2) 在 SFT 权重上挂新的 DPO LoRA（adapter 关闭时 = reference SFT 模型）
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(merged, lora_config)
    model.print_trainable_parameters()

    pad_id = tokenizer.pad_token_id
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        max_grad_norm=1.0,  # 梯度裁剪，防止 DPO 梯度爆炸
        fp16=False,
        bf16=torch.cuda.is_available(),  # RTX 3060 (Ampere) 支持 bf16，比 fp16 稳定
        seed=42,
        report_to="none",
        remove_unused_columns=False,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: dpo_collator(batch, pad_id),
        beta=0.1,
    )

    print("[DPO] Start DPO training...")
    dpo_trainer.train()

    best_dir = os.path.join(OUT_DIR, "best")
    dpo_trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n[DPO] Done! DPO LoRA adapter saved -> {best_dir}")


if __name__ == "__main__":
    train_dpo()
