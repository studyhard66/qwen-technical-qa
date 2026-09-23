# -*- coding: utf-8 -*-
"""评估工具：PPL、生成回复、训练问题诊断"""
import json
import math
import torch
import numpy as np
from typing import List, Dict


def compute_ppl(model, tokenizer, texts: List[str], max_length=2048,
                device="cuda", batch_size=4) -> float:
    """批量计算 PPL"""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=max_length, padding=True).to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            outputs = model(**enc, labels=labels)
            loss = outputs.loss
            n_tokens = (labels != -100).sum().item()
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

    return math.exp(total_nll / max(total_tokens, 1))


def generate_response(model, tokenizer, prompt: str, max_new_tokens=512,
                      device="cuda", temperature=0.7) -> str:
    """生成单条回复"""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def diagnose_training(train_losses: List[float], eval_losses: List[float]) -> Dict:
    """训练问题诊断"""
    result = {"issues": [], "suggestions": []}

    if len(train_losses) < 2:
        result["issues"].append("数据点不足，无法诊断")
        return result

    train_final = float(np.mean(train_losses[-3:]))
    eval_final = float(np.mean(eval_losses[-3:]))
    train_start = float(np.mean(train_losses[:3]))
    eval_start = float(np.mean(eval_losses[:3]))

    if train_final > train_start * 0.9:
        result["issues"].append("训练 loss 下降不明显，可能未收敛")
        result["suggestions"].append("增大学习率或增加训练轮数；检查数据和标签")

    if eval_final > eval_start and train_final < train_start:
        gap = eval_final - train_final
        result["issues"].append(f"验证 loss 上升，疑似过拟合 (gap={gap:.4f})")
        result["suggestions"].extend([
            "增加数据量或数据增强", "增大 LoRA dropout", "降低学习率",
            "启用早停", "减小 LoRA rank",
        ])

    if train_final > 2.0:
        result["issues"].append(f"训练 loss 仍较高 ({train_final:.4f})，疑似欠拟合")
        result["suggestions"].extend(["增加训练轮数", "增大学习率", "增大 LoRA rank"])

    train_std = float(np.std(train_losses[-10:])) if len(train_losses) >= 10 else 0
    if train_std > 0.5:
        result["issues"].append(f"训练 loss 震荡 (std={train_std:.4f})")
        result["suggestions"].extend([
            "降低学习率", "增大 batch size", "启用梯度裁剪", "检查数据质量",
        ])

    if any(math.isnan(l) or math.isinf(l) for l in train_losses):
        result["issues"].append("出现 NaN/Inf loss，梯度爆炸")
        result["suggestions"].extend([
            "降低学习率", "启用梯度裁剪", "检查数据异常值",
        ])

    if not result["issues"]:
        result["issues"].append("训练正常，无明显问题")

    return result
