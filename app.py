#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
app.py — Qwen1.5-1.8B 技术问答模型 FastAPI 部署

启动方式:
    python app.py
    # 或
    uvicorn app:app --host 0.0.0.0 --port 8000

环境变量:
    MODEL_VERSION  选择加载的模型: dpo(默认) / sft / base
"""
import os
import time
import warnings
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen1.5-1.8B"
MODEL_VERSION = os.getenv("MODEL_VERSION", "dpo").lower()

ADAPTER_PATHS = {
    "sft": os.path.join(ROOT, "models", "sft", "best"),
    "dpo": os.path.join(ROOT, "models", "dpo", "best"),
}

# 全局状态：模型只在启动时加载一次
state = {"model": None, "tokenizer": None, "device": "cpu"}

def load_pipeline(version: str):
    """加载基座模型并按级联方式挂 LoRA adapter（与 stage4_inference.py 一致）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_path = ADAPTER_PATHS.get(version)

    if version != "base":
        if not adapter_path or not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"模型 {version} 的 adapter 不存在: {adapter_path}，请先完成对应阶段训练"
            )

    tokenizer_source = adapter_path if (version != "base" and os.path.exists(adapter_path)) else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "sdpa" if device == "cuda" else "eager"
    config = AutoConfig.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True
    )
    config._attn_implementation = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        config=config,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )

    if version in ("sft", "dpo"):
        # DPO adapter 的基线是 SFT：先挂 SFT 并合并，再挂 DPO
        if version == "dpo" and os.path.exists(ADAPTER_PATHS["sft"]):
            model = PeftModel.from_pretrained(model, ADAPTER_PATHS["sft"], local_files_only=True)
            model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)

    model.eval()
    return model, tokenizer, device

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Server] 正在加载模型 ({MODEL_VERSION}) ...")
    t0 = time.time()
    model, tokenizer, device = load_pipeline(MODEL_VERSION)
    state["model"], state["tokenizer"], state["device"] = model, tokenizer, device
    print(f"[Server] 模型加载完成，设备: {device}，耗时 {time.time() - t0:.1f}s")
    yield
    print("[Server] 服务关闭")

app = FastAPI(title="Qwen1.5-1.8B 技术问答 API", version="1.0.0", lifespan=lifespan)

# 允许跨域（便于以后接独立前端 / 在线 Demo 页面调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="技术问题")
    max_new_tokens: int = Field(256, ge=16, le=512)
    temperature: float = Field(0.7, ge=0.1, le=1.5)
    top_p: float = Field(0.9, ge=0.1, le=1.0)

class ChatResponse(BaseModel):
    answer: str
    model: str
    device: str
    latency_s: float
    n_tokens: int

@app.get("/health")
def health():
    ready = state["model"] is not None
    return {"status": "ready" if ready else "loading", "model": MODEL_VERSION,
            "device": state["device"]}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    model, tokenizer, device = state["model"], state["tokenizer"], state["device"]
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    prompt = f"<|im_start|>user\n{req.question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 收集 stop tokens，防止 ChatML 模板泄露
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
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=True,
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids[0] if eos_ids else tokenizer.eos_token_id,
        )
    latency = time.time() - start

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    for stop in ["<|im_end|>", "<|endoftext|>"]:
        if stop in text:
            text = text.split(stop)[0]

    return ChatResponse(
        answer=text.strip(),
        model=MODEL_VERSION,
        device=device,
        latency_s=round(latency, 3),
        n_tokens=len(generated),
    )

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen1.5-1.8B 技术问答</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Segoe UI", sans-serif; background: #f5f7fa;
         height: 100vh; display: flex; flex-direction: column; }
  header { background: #1f2937; color: #fff; padding: 14px 24px; }
  header h1 { font-size: 18px; font-weight: 600; }
  header p { font-size: 12px; color: #9ca3af; margin-top: 2px; }
  #chat { flex: 1; overflow-y: auto; padding: 24px; max-width: 860px; width: 100%; margin: 0 auto; }
  .msg { margin-bottom: 16px; display: flex; }
  .msg.user { justify-content: flex-end; }
  .bubble { max-width: 80%; padding: 10px 14px; border-radius: 10px; font-size: 14px;
            line-height: 1.7; white-space: pre-wrap; word-break: break-word; }
  .user .bubble { background: #3b82f6; color: #fff; }
  .bot .bubble { background: #fff; border: 1px solid #e5e7eb; }
  .meta { font-size: 11px; color: #9ca3af; margin-top: 4px; }
  footer { padding: 16px 24px; background: #fff; border-top: 1px solid #e5e7eb; }
  .bar { max-width: 860px; margin: 0 auto; display: flex; gap: 8px; }
  #q { flex: 1; padding: 10px 14px; border: 1px solid #d1d5db; border-radius: 8px;
       font-size: 14px; resize: none; height: 44px; }
  button { padding: 0 22px; background: #3b82f6; color: #fff; border: none;
           border-radius: 8px; font-size: 14px; cursor: pointer; }
  button:disabled { background: #9ca3af; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>Qwen1.5-1.8B 技术问答模型</h1>
  <p>LoRA SFT + 手写 DPO 微调 · FastAPI 部署</p>
</header>
<div id="chat">
  <div class="msg bot"><div class="bubble">你好！我是技术问答助手，试着问我一个技术问题吧，例如：
「MySQL 索引为什么用 B+ 树？」</div></div>
</div>
<footer>
  <div class="bar">
    <textarea id="q" placeholder="输入技术问题，Enter 发送…"></textarea>
    <button id="send">发送</button>
  </div>
</footer>
<script>
const chat = document.getElementById('chat');
const q = document.getElementById('q');
const btn = document.getElementById('send');
function add(role, text, meta) {
  const row = document.createElement('div');
  row.className = 'msg ' + role;
  row.innerHTML = '<div><div class="bubble"></div>' +
    (meta ? '<div class="meta">' + meta + '</div>' : '') + '</div>';
  row.querySelector('.bubble').textContent = text;
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}
async function send() {
  const question = q.value.trim();
  if (!question) return;
  add('user', question);
  q.value = '';
  btn.disabled = true;
  const t0 = performance.now();
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.status);
    const d = await r.json();
    add('bot', d.answer,
      `模型: ${d.model} · 设备: ${d.device} · ${d.latency_s}s · ${d.n_tokens} tokens`);
  } catch (e) {
    add('bot', '出错了：' + e.message);
  } finally {
    btn.disabled = false;
  }
}
btn.onclick = send;
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
