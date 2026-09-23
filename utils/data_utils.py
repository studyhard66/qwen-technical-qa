# -*- coding: utf-8 -*-
"""数据工具：ChatML 格式化、JSONL 读写、PPL 计算、规则校验"""
import json
import math
import torch
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def save_jsonl(items: List[Dict], path: str):
    """保存 JSONL 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def to_chatml_messages(question: str, answer: str, context: str = "",
                       system_prompt: str = None) -> Dict:
    """将问答对转为 ChatML messages 格式"""
    from config.prompt_templates import SYSTEM_PROMPT, SFT_USER_TEMPLATE, SFT_USER_NO_CONTEXT

    sys = system_prompt or SYSTEM_PROMPT
    if context:
        user_content = SFT_USER_TEMPLATE.format(context=context, question=question)
    else:
        user_content = SFT_USER_NO_CONTEXT.format(question=question)

    return {
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    }


def compute_perplexity(model, tokenizer, texts: List[str],
                       max_length: int = 2048, device: str = "cuda") -> float:
    """计算模型在文本列表上的平均困惑度（PPL = exp(平均NLL)）"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_length
            ).to(device)
            if encoding["input_ids"].shape[1] < 2:
                continue
            labels = encoding["input_ids"].clone()
            outputs = model(**encoding, labels=labels)
            loss = outputs.loss
            n_tokens = labels.shape[1]
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def quality_rule_check(item: Dict) -> bool:
    """规则校验：返回 True 表示通过"""
    q = item.get("question", "").strip()
    a = item.get("answer", "").strip()
    if not q or not a:
        return False
    if len(a) < 20:
        return False
    has_structure = any(
        s in a for s in ["```", "1.", "2.", "- ", "|", "```python",
                         "```bash", "```yaml", "```sql", "```jsx", "```java"]
    )
    if not has_structure:
        return False
    total_len = len(q) + len(a)
    if total_len > 2048:
        return False
    return True
