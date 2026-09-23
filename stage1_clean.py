#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage1_clean.py — 数据清洗与预处理
1. MinHash 模糊去重（Jaccard 阈值 0.95，比之前宽松，避免误删同类问答）
2. 质量筛选：长度合理 + answer 含结构化内容
3. 8:1:1 划分 train/val/test，输出统计报告
"""
import json
import os
import hashlib
import random
from collections import defaultdict, Counter

import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = "D:/SFT"
RAW = os.path.join(ROOT, "data/raw_qa.jsonl")
OUT = os.path.join(ROOT, "data/processed")
os.makedirs(OUT, exist_ok=True)


def tokenize(text):
    """character 3-gram shingles"""
    tokens = set()
    for i in range(len(text) - 2):
        tokens.add(text[i:i + 3])
    return tokens


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


N_BINS = 16


def minhash_signature(tokens):
    """对每个 band 取 min hash"""
    sig = []
    for b in range(N_BINS):
        min_val = float("inf")
        for t in tokens:
            val = int(hashlib.md5(f"band{b}:{t}".encode()).hexdigest(), 16)
            min_val = min(min_val, val)
        sig.append(min_val)
    return sig


def minhash_estimate(sig_a, sig_b):
    return np.mean(np.array(sig_a) == np.array(sig_b))


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "question" in obj and "answer" in obj:
                    items.append(obj)
            except json.JSONDecodeError:
                continue
    return items


def clean_data(items):
    """去重 + 质量筛选"""
    total_before = len(items)
    print(f"[STAGE1] Raw data: {total_before} items")

    # --- 1. MinHash 去重 ---
    print("[STAGE1] Running MinHash deduplication (threshold=0.95)...")
    signatures = [minhash_signature(tokenize(it["answer"])) for it in items]

    bins = defaultdict(list)
    for i, sig in enumerate(signatures):
        bins[sig[0]].append(i)

    dup_indices = set()
    checked = 0
    for group in bins.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ii, jj = group[i], group[j]
                if ii in dup_indices or jj in dup_indices:
                    continue
                est = minhash_estimate(signatures[ii], signatures[jj])
                if est < 0.85:
                    continue
                sim = jaccard(tokenize(items[ii]["answer"]),
                              tokenize(items[jj]["answer"]))
                checked += 1
                if sim > 0.95:
                    # 保留更长的
                    if len(items[jj]["answer"]) > len(items[ii]["answer"]):
                        dup_indices.add(ii)
                    else:
                        dup_indices.add(jj)

    print(f"[STAGE1] Checked {checked} pairs, removed {len(dup_indices)} duplicates")
    items = [it for idx, it in enumerate(items) if idx not in dup_indices]
    after_dedup = len(items)
    print(f"[STAGE1] After dedup: {after_dedup} items")

    # --- 2. 质量筛选 ---
    print("[STAGE1] Quality filtering...")
    filtered = []
    for item in items:
        q = item["question"].strip()
        a = item["answer"].strip()
        if not q or not a:
            continue
        if len(a) < 20:
            continue
        # 答案需要有结构化内容
        has_structure = any(s in a for s in [
            "```", "1.", "2.", "- ", "|", "```python", "```bash",
            "```yaml", "```sql", "```jsx", "```java", "```http",
            "```text", "```dockerfile",
        ])
        if not has_structure:
            continue
        if len(q) + len(a) > 2048:
            continue
        filtered.append(item)

    after_filter = len(filtered)
    print(f"[STAGE1] After quality filter: {after_filter} items")
    return filtered, total_before, after_dedup, after_filter


def split_and_save(items, report_path, before, after_dedup, after_filter):
    """8:1:1 划分 train/val/test"""
    random.shuffle(items)
    n = len(items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(os.path.dirname(report_path), f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[STAGE1] Saved {len(data)} items -> {path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("数据清洗统计报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"原始数据量:      {before}\n")
        f.write(f"去重后:          {after_dedup} (去除 {before - after_dedup} 条)\n")
        f.write(f"质量筛选后:      {after_filter} (去除 {after_dedup - after_filter} 条)\n\n")
        f.write(f"训练集 (train):  {len(train)} 条\n")
        f.write(f"验证集 (val):    {len(val)} 条\n")
        f.write(f"测试集 (test):   {len(test)} 条\n\n")
        # 答案长度分布
        counter = Counter()
        for item in train:
            l = len(item["answer"])
            if l < 100:
                counter["<100"] += 1
            elif l < 300:
                counter["100-300"] += 1
            elif l < 600:
                counter["300-600"] += 1
            else:
                counter[">600"] += 1
        f.write("训练集答案长度分布:\n")
        for k, v in sorted(counter.items()):
            f.write(f"  {k}: {v} 条\n")

    print(f"[STAGE1] Report -> {report_path}")


def main():
    print("=" * 60)
    print("STAGE 1: Data Cleaning & Preprocessing")
    print("=" * 60)

    items = load_jsonl(RAW)
    filtered, before, after_dedup, after_filter = clean_data(items)

    if not filtered:
        print("[STAGE1] ERROR: No data after cleaning!")
        return

    report_path = os.path.join(OUT, "data_report.txt")
    split_and_save(filtered, report_path, before, after_dedup, after_filter)
    print("[STAGE1] Done.\n")


if __name__ == "__main__":
    main()
