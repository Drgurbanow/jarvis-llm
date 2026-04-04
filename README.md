# 🤖 JARVIS-LLM

**Fine-tuned Phi-3.5-mini & Phi-4-mini to replicate JARVIS personality with RAG capability**

> *"At your service, Sir."*  
> — JARVIS, Iron Man

---

## 📌 Overview

This project fine-tunes Microsoft's Phi-3.5-mini and Phi-4-mini (3.8B parameters) 
using QLoRA to replicate the JARVIS AI assistant personality from the Iron Man universe, 
while preserving the base model's RAG and reasoning capabilities.

**Goals:**
- Style transfer: make the model respond like JARVIS
- Preserve RAG context utilization from base model
- Evaluate trade-offs between style and capability regression

---

## 🏗️ Architecture

architecture_diagram.png

**Full pipeline:**
1. Filter Alpaca-52K dataset by 7 categories
2. Generate JARVIS-style dialogues via DeepSeek API (async)
3. Convert Alpaca format → Phi Chat Template
4. QLoRA fine-tuning (attention-only, rank=16)
5. RAG pipeline for context-aware responses
6. LLM-as-a-Judge evaluation (DeepSeek v3.2 + Kimi 2.5)

---

## 📊 Results

| Model | Style | Context Util | Reasoning | Hallucination |
|-------|-------|--------------|-----------|---------------|
| Phi-3.5-mini Base | 5.3 | 9.0 | 7.5 | 2.2% |
| Phi-3.5-mini FT | 8.3 | 7.5 | 7.2 | ~6% |
| Phi-4-mini Base | 5.2 | 9.5 | 9.0 | 2.2% |
| Phi-4-mini FT 4-bit | 8.4 | 7.8 | 7.3 | 5% |
| Phi-4-mini FT 8-bit | 8.4 | 8.0 | 8.3 | 4% |

*Evaluated on 45 examples using LLM-as-a-Judge (DeepSeek v3.2 + Kimi 2.5)*  
*Variance: ±0.5 points*

---

## 🔍 Key Findings

**✅ Primary objective achieved:**
- Style improved +57% for Phi-3.5-mini (5.3 → 8.3)
- Style improved +62% for Phi-4-mini (5.2 → 8.4)

**⚠️ Expected trade-offs (catastrophic forgetting):**
- Context Utilization regressed after fine-tuning on both models
- Training focused on style — RAG behavior inherited from base model

**🔬 Key discovery — quantization sensitivity:**
- Phi-4-mini shows meaningful improvement at 8-bit vs 4-bit:
  - Reasoning: 7.3 → 8.3 (+1.0)
- Phi-3.5-mini shows no significant difference between 4-bit and 8-bit
- Suggests Phi-4-mini's more complex architecture is more sensitive 
  to quantization precision

**📉 Hallucination analysis:**
- Strict rate (factual errors only): ~2-4%
- Broad rate (including arithmetic errors): ~6-7%
- Both within acceptable range for 3.8B models

---

## 📁 Project Structure
jarvis-llm/
├── README.md
├── architecture_diagram.png
├── data_pipeline/
│   ├── filter_alpaca.py        # Filter Alpaca-52K by 7 categories
│   ├── generate_dialogues.py   # Async DeepSeek API dialogue generation
│   └── format_to_chat.py       # Alpaca → Phi Chat Template conversion
├── training/
│   ├── train.py                # QLoRA fine-tuning script
│   └── config.yaml             # Hyperparameters
├── rag_pipeline/
│   └── rag_inference.py        # RAG pipeline implementation
├── evaluation/
│   ├── test_cases.jsonl        # 45 test examples
│   ├── generate_responses.py   # Batch inference script
│   ├── results.csv    # Raw scores — Phi-3.5\4-mini FT
│   └── summary.md              # Analysis and conclusions
├── requirements.txt
└── LICENSE
---

## ⚙️ Setup
```bash
# Clone repository
git clone https://github.com/Drgurbanow/jarvis-llm.git
cd jarvis-llm

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

**Run fine-tuned model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model + adapter
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct")
model = PeftModel.from_pretrained(model, "YOUR_HF_USERNAME/jarvis-phi35-lora")
model.eval()
# Generate response
inputs = tokenizer("Jarvis, what is the status of the Mark XLII?", 
                   return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

**Run evaluation:**
```bash
cd evaluation
python generate_responses.py
```

---

## 🗂️ Dataset & Models

| Resource | Link |
|----------|------|
| Dataset | 🤗 HuggingFace *(coming soon)* |
| Phi-3.5-mini Adapter | 🤗 [Drgurbanow/jarvis-phi35-lora](coming soon) |
| Phi-4-mini Adapter | 🤗 [Drgurbanow/jarvis-phi4-lora](coming soon) |

---

## 🛠️ Technical Details

| Parameter | Value |
|-----------|-------|
| Base models | Phi-3.5-mini-instruct, Phi-4-mini-instruct |
| Parameters | 3.8B |
| Method | QLoRA (4-bit NF4) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | Attention only (no MLP) |
| Dataset size | 5,905 dialogues |
| Training epochs | 3 |
| Hardware | NVIDIA RTX 5060|

---

## 📝 Limitations

- 45-example evaluation set (student project scope)
- Manual LLM-as-a-Judge scoring
- Style fine-tuning only — RAG not explicitly trained
- Arithmetic reasoning limited by base model capacity

---

## 🔗 References

- [Phi-3.5 Technical Report](https://arxiv.org/abs/2404.14219)
- [Phi-4-mini Technical Report](https://arxiv.org/abs/2503.01743)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Vectara Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard)

---

---

# 🤖 JARVIS-LLM (中文版本)

**基于 Phi-3.5-mini 和 Phi-4-mini 微调的 JARVIS 风格 AI 助手**

---

## 📌 项目简介

本项目使用 QLoRA 对微软的 Phi-3.5-mini 和 Phi-4-mini（3.8B 参数）进行微调，
使模型具备钢铁侠宇宙中 JARVIS AI 助手的对话风格，
同时保留基础模型的 RAG 和推理能力。

**项目目标：**
- 风格迁移：使模型像 JARVIS 一样回应
- 保留基础模型的 RAG 上下文利用能力
- 评估风格与能力退化之间的权衡

---

## 📊 实验结果

| 模型 | 风格 | 上下文利用 | 推理 | 幻觉率 |
|------|------|-----------|------|--------|
| Phi-3.5-mini 基础版 | 5.3 | 9.0 | 7.5 | 2.2% |
| Phi-3.5-mini 微调版 | 8.3 | 7.5 | 7.2 | ~6% |
| Phi-4-mini 基础版 | 5.2 | 9.5 | 9.0 | 2.2% |
| Phi-4-mini 微调版 4-bit | 8.4 | 7.8 | 7.3 | 5% |
| Phi-4-mini 微调版 8-bit | 8.4 | 8.0 | 8.3 | 4% |

---

## 🔍 主要发现

**✅ 主要目标达成：**
- Phi-3.5-mini 风格提升 +57%（5.3 → 8.3）
- Phi-4-mini 风格提升 +62%（5.2 → 8.4）

**🔬 重要发现 — 量化敏感性：**
- Phi-4-mini 在 8-bit 量化下表现明显优于 4-bit
- 推理能力：7.3 → 8.3（+1.0）
- Phi-3.5-mini 在两种量化精度下无显著差异
- 说明 Phi-4-mini 更复杂的架构对量化精度更敏感

**⚠️ 预期的权衡（灾难性遗忘）：**
- 微调后上下文利用率有所下降
- 这是风格专注训练的预期结果

---

## ⚙️ 安装
```bash
git clone https://github.com/Drgurbanow/jarvis-llm.git
cd jarvis-llm
pip install -r requirements.txt
```

---

## 🛠️ 技术参数

| 参数 | 数值 |
|------|------|
| 基础模型 | Phi-3.5-mini, Phi-4-mini |
| 参数量 | 3.8B |
| 训练方法 | QLoRA (4-bit NF4) |
| LoRA rank | 16 |
| 数据集大小 | 5,905 条对话 |
| 训练轮次 | 3 |
| 硬件 | NVIDIA RTX 5060|

---

## 🚀 使用方法

详见英文版本的 Usage 部分。
(See English version for detailed usage instructions.)
