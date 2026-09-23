# -*- coding: utf-8 -*-
"""结构化 Prompt 模板集合"""

SYSTEM_PROMPT = (
    "你是一个严谨的技术文档问答助手。\n"
    "你的回答必须：\n"
    "1. 只基于给定的资料回答，不编造不确定的内容；\n"
    "2. 分步骤、结构化，关键结论可溯源到资料原文；\n"
    "3. 使用 Markdown 格式，代码用 ``` 包裹；\n"
    "4. 简洁准确，避免泛泛而谈；\n"
    "5. 遇到资料中没有的信息，明确说明\"资料中未提及\"。"
)

SFT_USER_TEMPLATE = (
    "资料：\n{context}\n\n"
    "问题：{question}\n\n"
    "请基于上述资料，给出分步骤、可溯源、简洁的回答。"
)

SFT_USER_NO_CONTEXT = "问题：{question}\n\n请给出专业、准确的回答。"

QA_GEN_SYSTEM = "你是一个技术文档问答对生成专家，需要从给定的文档块中抽取或生成高质量的问答对。"

QA_GEN_USER_TEMPLATE = (
    "请从以下技术文档块中生成 3 个不同角度的问答对。\n"
    "要求：问题必须能从文档中找到答案；答案可溯源；问题类型多样。\n"
    "输出 JSON 数组：[{{\"question\": \"...\", \"answer\": \"...\"}}]\n\n"
    "文档块：\n{chunk}"
)

QUALITY_CHECK_SYSTEM = "你是一个数据质量审核员，对问答对进行多维度打分（1-5分）。"

QUALITY_CHECK_USER_TEMPLATE = (
    "维度：正确性、相关性、完整性、可读性、无幻觉、格式合规\n"
    "问题：{question}\n答案：{answer}\n"
    "输出 JSON：{{\"correctness\":x,\"relevance\":x,\"completeness\":x,"
    "\"readability\":x,\"no_hallucination\":x,\"format\":x,\"total\":x}}"
)

REWARD_TEMPLATE = (
    "请对以下回答进行偏好评分（0-10分），维度：专业性、准确性、完整性、格式规范、用户意图满足度。\n"
    "问题：{question}\n回答：{answer}\n只输出一个数字。"
)


def format_chatml(system: str, user: str, assistant: str = None) -> str:
    """格式化为 Qwen ChatML 格式"""
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
    if assistant is not None:
        parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>\n")
    return "".join(parts)


def format_messages(messages: list) -> str:
    """将 messages 列表格式化为 ChatML"""
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return text
