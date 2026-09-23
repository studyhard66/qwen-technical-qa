# Qwen1.5-1.8B 技术问答模型：数据工程 + LoRA SFT + DPO 偏好对齐全流程

从 0 到 1 完成大模型微调闭环：**数据生成/清洗 → LoRA SFT → 手写 DPO 偏好对齐 → 推理评估**。
不依赖 TRL，DPO 损失基于 transformers Trainer 手写实现，单卡 RTX 3060 (12GB) 可全流程复现。

## 项目架构

```
数据流水线
generate_raw_data.py   生成技术问答种子+模板数据
        ↓  1047 条原始 QA
stage1_clean.py        MinHash 去重 + 质量过滤 + 8:1:1 划分
        ↓  448 条 → train 358 / val 44 / test 46
stage2_sft.py          LoRA SFT（ChatML，只对 answer 计 loss，早停监控 PPL）
        ↓  models/sft/best（LoRA adapter）
stage3_dpo.py          手写 DPO：合并 SFT adapter → 挂新 LoRA
        ↓  models/dpo/best（LoRA adapter）
stage4_inference.py    Base / SFT / DPO 三方生成对比
        ↓  models/dpo/best（LoRA adapter）
app.py                 FastAPI 服务化：/api/chat 接口 + 网页 Demo
}```
## 训练配置与结果

### Stage 2: SFT（LoRA）

| 超参 | 值 |
|------|-----|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| target_modules | q/k/v/o + gate/up/down_proj |
| 学习率 | 2e-4, cosine, warmup 5% |
| batch | 4 × 8 (effective 32) |
| epochs | 5（早停 patience=3，监控验证 PPL） |
| 精度 | fp16 + gradient checkpointing |

**结果**：train loss 1.83 → 0.16，验证 PPL 1.80 → **1.26**，第 5 轮触发早停，训练 354s。

### Stage 3: DPO（手写实现）

| 超参 | 值 |
|------|-----|
| beta | 0.1 |
| 学习率 | 5e-5, cosine |
| batch | 1 × 16 |
| 精度 | bf16 + max_grad_norm=1.0 |
| reference 模型 | `disable_adapter()` 动态计算，零额外显存 |

**偏好对构造**：chosen = 原始结构化回答（含代码块/序号）；rejected = 截断+去结构化退化回答。

**结果**：train loss 0.678 → 0.010（初始值 ≈ ln2，符合 DPO 理论起点），val loss 0.059 → 0.034，训练 278s。

### 生成效果对比（同一问题）

| 模型 | 回答风格 |
|------|---------|
| Base | 纯文字、啰嗦、概念混乱 |
| SFT | 条目化、简洁，学会结构化输出 |
| DPO | 直接以代码块开头 + 解释，最贴近目标风格 |

详细输出见 `inference_report.json`。

## 快速开始

```powershell
# 1. 环境
pip install -r requirements.txt
# pip install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121

# 2. 数据
python generate_raw_data.py
python stage1_clean.py

# 3. 训练
python stage2_sft.py     # ~6 min
python stage3_dpo.py     # ~5 min

# 4. 评估
python stage4_inference.py
```
> 模型权重（Qwen/Qwen1.5-1.8B）首次运行自动从 HF 下载，国内可设 `$env:HF_ENDPOINT="https://hf-mirror.com"` 。

## 服务部署（FastAPI）

将 DPO 模型封装为 HTTP 服务，模型启动时只加载一次，浏览器打开即可对话：

```powershell
python app.py    # 默认加载 DPO 模型，加载约 10s
```

| 地址 | 用途 |
|------|------|
| http://localhost:8000 | 网页聊天 Demo |
| http://localhost:8000/docs | 自动接口文档（Swagger UI，可在线调试） |
| http://localhost:8000/health | 健康检查 |

接口调用（PowerShell）：

```powershell
curl.exe -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"question\": \"MySQL 索引为什么用 B+ 树？\"}"
```

返回 `answer`（回答）、`latency_s`（延迟）、`n_tokens`（生成 token 数）等字段。

切换加载的模型版本（环境变量 `MODEL_VERSION`，默认 `dpo`）：

```powershell
$env:MODEL_VERSION="sft"; python app.py   # 可选 dpo / sft / base
```

## 目录结构

```
├── generate_raw_data.py
├── stage1_clean.py
├── stage2_sft.py
├── stage3_dpo.py
├── stage4_inference.py
├── app.py                  # FastAPI 服务（接口 + 网页 Demo）
├── config/prompt_templates.py
├── utils/
├── data/
├── models/
├── docs/
└── requirements.txt
```
## 踩坑记录（工程经验）

1. **FlashAttention-2 在 Windows 不可用**：flash-attn 编译依赖 Linux 工具链；用 `AutoConfig` 显式设 `_attn_implementation="sdpa"`（PyTorch 内置等价实现）替代。
2. **transformers 5.x API 变更**：`warmup_ratio`/`logging_dir` 已移除（前者并入 `warmup_steps` 支持浮点比例，后者由 `output_dir` 自动管理）。
3. **TRL 与 torch 2.5.1 不兼容**：新版 TRL 依赖 `FSDPModule`（torch 2.6+），降级会破坏环境 → 手写 DPO 损失，反而加深了对算法的理解。
4. **LoRA adapter 不能直接 from_pretrained**：需先加载基座再 `PeftModel.from_pretrained`；DPO adapter 的基线是 SFT，推理加载需级联（base → merge SFT → 挂 DPO adapter）。
5. **fp16 下 DPO 梯度 nan**：loss 骤降导致数值溢出 → 改 bf16（Ampere+ 支持）+ 梯度裁剪。
6. **DPO 显存优化**：reference logps 用 `disable_adapter()` 在同一模型上计算，避免加载第二份模型（省 ~3.6GB）。
7. **ChatML 模板泄露**：推理时需将 `<|im_end|>` 设为 eos token 并在解码后截断，否则模型生成完回答会继续编造下一轮对话。

## 已知局限

- 数据为模板合成为主（358 条训练样本），知识密度有限，回答存在事实性错误
- 1.8B 模型容量小，适合流程演示而非生产使用
- 缺少 BLEU/ROUGE 等定量评估（PPL 之外）

## 技术栈

PyTorch 2.5.1+cu121 · transformers 5.17.0 · peft 0.21.0 · datasets · NumPy