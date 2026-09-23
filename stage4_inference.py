#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage4_inference.py — 推理 + FlashAttention 性能分析
1. 加载基座 / SFT / DPO 模型，对比生成效果
2. 对比 FlashAttention-2 vs Eager 注意力的推理性能（延迟、吞吐、显存）
3. 输出性能分析报告
"""
import json
import os
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

warnings.filterwarnings("ignore")

ROOT = "D:/SFT"
BASE_MODEL = "Qwen/Qwen1.5-1.8B"
SFT_MODEL = os.path.join(ROOT, "models/sft/best")
DPO_MODEL = os.path.join(ROOT, "models/dpo/best")

TEST_QUESTIONS = [
    "Kubernetes 中 Pod 和 Deployment 的区别是什么？",
    "MySQL 索引为什么用 B+ 树？",
    "React 中 useEffect 的依赖数组有什么作用？",
    "Python 中 *args 和 **kwargs 的区别？",
]


def load_model(model_path, use_flash=True, device="cuda"):
    """加载模型，可选 FlashAttention（强制用本地缓存，不联网）"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if os.path.exists(model_path) else BASE_MODEL,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "sdpa" if (use_flash and torch.cuda.is_available()) else "eager"
    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    config._attn_implementation = attn_impl
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else None,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        local_files_only=True,
    )

    # 如果是 adapter 路径，加载 LoRA adapter
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # DPO adapter 的基线是 SFT：先挂 SFT adapter 并合并，再挂 DPO adapter
        if os.path.abspath(model_path) == os.path.abspath(DPO_MODEL) and os.path.exists(SFT_MODEL):
            model = PeftModel.from_pretrained(model, SFT_MODEL, local_files_only=True)
            model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, model_path, local_files_only=True)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, question, max_new_tokens=256, device="cuda"):
    """生成单条回复"""
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 收集 eos / im_end token id，作为 stop tokens
    eos_ids = []
    for tok_str in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids[0] if eos_ids else tokenizer.eos_token_id,
        )
    latency = time.time() - start

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    # 截断到第一个 im_end（防止模板泄露）
    for stop in ["<|im_end|>", "<|endoftext|>"]:
        if stop in text:
            text = text.split(stop)[0]
    n_tokens = len(generated)
    return text, latency, n_tokens


def benchmark_inference(model, tokenizer, device="cuda", n_runs=5):
    """推理性能基准测试"""
    results = {"latencies": [], "throughputs": [], "vram_mb": 0}

    # 预热
    generate(model, tokenizer, TEST_QUESTIONS[0], max_new_tokens=64, device=device)

    for q in TEST_QUESTIONS[:n_runs]:
        text, latency, n_tokens = generate(model, tokenizer, q, max_new_tokens=256, device=device)
        results["latencies"].append(latency)
        results["throughputs"].append(n_tokens / latency if latency > 0 else 0)

    if torch.cuda.is_available():
        results["vram_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return results


def compare_flash_attention(device="cuda"):
    """对比 FlashAttention-2 vs Eager"""
    print("\n" + "=" * 60)
    print("FlashAttention vs Eager 推理性能对比")
    print("=" * 60)

    comparison = {}
    for use_flash in [True, False]:
        name = "FlashAttention-2" if use_flash else "Eager"
        print(f"\n[Bench] Loading model with {name}...")
        try:
            model, tokenizer = load_model(BASE_MODEL, use_flash=use_flash, device=device)
            results = benchmark_inference(model, tokenizer, device=device)

            avg_latency = sum(results["latencies"]) / len(results["latencies"])
            avg_tput = sum(results["throughputs"]) / len(results["throughputs"])
            comparison[name] = {
                "avg_latency_s": round(avg_latency, 3),
                "avg_throughput_tok_s": round(avg_tput, 2),
                "vram_mb": round(results["vram_mb"], 1),
            }
            print(f"  {name}: latency={avg_latency:.3f}s, "
                  f"throughput={avg_tput:.2f} tok/s, VRAM={results['vram_mb']:.1f}MB")

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name} failed: {e}")
            comparison[name] = {"error": str(e)}

    # 计算加速比
    if "FlashAttention-2" in comparison and "Eager" in comparison:
        fa = comparison["FlashAttention-2"]
        ea = comparison["Eager"]
        if "avg_latency_s" in fa and "avg_latency_s" in ea and ea["avg_latency_s"] > 0:
            speedup = ea["avg_latency_s"] / fa["avg_latency_s"]
            comparison["speedup"] = round(speedup, 2)
            print(f"\n  FlashAttention 加速比: {speedup:.2f}x")

    return comparison


def compare_models(device="cuda"):
    """对比基座 / SFT / DPO 模型生成效果"""
    print("\n" + "=" * 60)
    print("基座 vs SFT vs DPO 生成效果对比")
    print("=" * 60)

    models_to_test = [("Base", BASE_MODEL)]
    if os.path.exists(SFT_MODEL):
        models_to_test.append(("SFT", SFT_MODEL))
    if os.path.exists(DPO_MODEL):
        models_to_test.append(("DPO", DPO_MODEL))

    results = {}
    for name, path in models_to_test:
        print(f"\n[{name}] Generating responses...")
        try:
            model, tokenizer = load_model(path, use_flash=True, device=device)
            responses = {}
            for q in TEST_QUESTIONS:
                text, _, _ = generate(model, tokenizer, q, device=device)
                responses[q] = text[:300]  # 截断展示
            results[name] = responses

            # 打印前两个问题的回答
            for q in TEST_QUESTIONS[:2]:
                print(f"\n  Q: {q}")
                print(f"  A: {responses[q][:200]}...")

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = {"error": str(e)}

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Inference] Device: {device}")

    print("\n[Inference] 已跳过 FlashAttention 性能对比（避免联网下载）")

    # 模型生成效果对比（Base / SFT / DPO 都用本地缓存的基座模型）
    model_comparison = compare_models(device)

    # 保存报告
    report = {
        "skipped": ["flash_attention_comparison (需联网，已跳过)"],
        "model_comparison_summary": {
            k: list(v.keys()) if isinstance(v, dict) else str(v)
            for k, v in model_comparison.items()
        },
    }
    report_path = os.path.join(ROOT, "inference_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[Inference] Report saved -> {report_path}")


if __name__ == "__main__":
    main()
