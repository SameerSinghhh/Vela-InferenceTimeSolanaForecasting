# 🧠 Inference-Time LLM Forecasting for Solana

> **Building the future of cryptocurrency prediction through memory-enhanced AI**

[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)](https://github.com)
[![Accuracy](https://img.shields.io/badge/Improvement-2.9pp_MAE-blue?style=for-the-badge)](https://github.com)
[![Model](https://img.shields.io/badge/LLM-OpenAI_o3--mini-orange?style=for-the-badge)](https://github.com)

---

## 🚀 What We Built

A **inference-time learning system** that predicts Solana (SOL) price movements using LLM memory—**no model training required**. Our AI learns from past predictions and market patterns in real-time.

```
💡 Traditional ML: Train weights → Deploy → Static predictions
🧠 Our Approach: Deploy → Learn at inference → Adaptive predictions
```

## 🎯 Key Innovation

<table>
<tr>
<td width="50%">

### 🔥 **Memory-Enhanced Predictions**
- **Short-term Memory (STM):** Recent prediction performance
- **Long-term Memory (LTM):** FAISS similarity search  
- **Real-time Learning:** Improves with each prediction

</td>
<td width="50%">

### 📊 **Proven Results**
- **2.9pp** lower prediction errors
- **Same trading returns** with higher accuracy
- **Maintained directional accuracy** while reducing errors

</td>
</tr>
</table>

---

## ⚡ Quick Start

### 🔧 Setup
```bash
# 1. Clone & install
git clone <repo-url>
pip install requests python-dotenv openai faiss-cpu pandas numpy

# 2. Add your API keys
cat > .env << EOF
OPENAI_API_KEY=your_key_here
SERP_API_KEY=your_key_here
SERP_BASE_URL=https://csearch.vela.partners/search
EOF

# 3. Run the system
python 05_predict_with_memory.py
```

### 📁 Project Structure
```
📦 inference_time_model/
├── 🎯 01_collect_data.py          # Price data pipeline
├── 📰 02_collect_news.py          # News scraping & AI summarization
├── 🤖 03_generate_predictions.py  # LLM predictions & reflections  
├── 🧠 04_embed_training_set.py    # FAISS memory system
├── ⚡ 05_predict_with_memory.py   # Memory-enhanced predictions
├── 📊 06_evaluate_predictions.py  # Performance evaluation
├── 📋 training_set.csv            # 20 weeks of training data
└── 🗄️ memory/                     # FAISS embeddings & index
```

---

## 🏆 Current Results

<div align="center">

### 📈 **Performance Comparison**

| Metric | Baseline | Memory-Enhanced | 🎯 Improvement |
|--------|----------|-----------------|----------------|
| **MAE (% points)** | 14.99pp | **12.11pp** | **-2.9pp** ✅ |
| **RMSE (% points)** | 16.89pp | **13.18pp** | **-3.7pp** ✅ |
| **Max Error** | 24.62pp | **19.00pp** | **-5.6pp** ✅ |

</div>

<div align="center">

### 🧠 **How Memory Works**

```mermaid
graph LR
    A[📰 Current News] --> B[🔍 FAISS Search]
    B --> C[📚 Similar Examples]
    C --> D[🤖 Enhanced Prompt]
    D --> E[🎯 Better Prediction]
    
    F[📊 Recent Performance] --> D
    
    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style D fill:#fff3e0
```

</div>

---

## 🔬 Technical Architecture

<table>
<tr>
<td width="60%">

### 🛠️ **System Components**
- **🎯 Data Pipeline:** 20 weeks of SOL price + news data
- **🧠 Memory System:** 1,536-dim embeddings via OpenAI
- **🔍 Similarity Search:** FAISS cosine similarity matching
- **📊 A/B Testing:** Memory vs baseline comparison

</td>
<td width="40%">

### 🔗 **APIs & Tools**
- **LLM:** OpenAI o3-mini
- **Search:** SERP API
- **Memory:** FAISS vectors
- **Data:** SOL price history

</td>
</tr>
</table>

### 💡 **Memory System in Action**

<details>
<summary><b>🔍 Click to see example memory context</b></summary>

```
🕐 SHORT-TERM MEMORY (Recent Performance):
- Week 2025-04-23: Predicted +10.5%, Actual -2.1% (❌ WRONG)
  💭 Reflection: "Too optimistic about market sentiment..."

🧠 LONG-TERM MEMORY (Similar Situations):
- Week 2025-02-12 (similarity: 0.816)
  📰 Context: "Strong ETF applications and media coverage..."
  📊 Result: Predicted +15.0%, Actual -13.9% ❌
  💡 Lesson: "ETF news doesn't always translate to gains..."

🎯 CURRENT PREDICTION:
Using lessons from memory to make smarter forecasts...
```

</details>

---

## 📋 **Training Dataset**

<div align="center">

| 📊 **Metric** | 📈 **Value** |
|:----------:|:----------:|
| **Training Weeks** | 20 weeks |
| **Date Range** | Jul 2024 - Apr 2025 |
| **Data Completeness** | 100% ✅ |
| **News Coverage** | 95% ✅ |
| **Prediction Accuracy** | 52.9% directional |

</div>

<details>
<summary><b>📋 View dataset schema</b></summary>

| Column | Description |
|--------|-------------|
| `target_date` | 📅 Prediction point |
| `target_price` | 💰 SOL price at prediction |
| `summarized_context` | 📰 AI-generated news summary |
| `predicted_price` | 🤖 LLM price forecast |
| `actual_price` | 📊 Real SOL price 7 days later |
| `llm_reasoning` | 🧠 AI's prediction logic |
| `llm_reflection` | 💭 AI's performance analysis |

</details>

---

## 🎯 **Next Steps**

<div align="center">

### 🚀 **Roadmap**

```
🔄 Pipeline Optimization     🧠 Enhanced Memory Algorithms     📊 Extended Testing
      ↓                            ↓                              ↓
  Faster processing          Better similarity search       Longer evaluation
      ↓                            ↓                              ↓
    Q1 2025                      Q2 2025                     Q2 2025
```

</div>

### 🎨 **Improvement Areas**
- 🔍 **Advanced Similarity Search:** Semantic clustering and weighted retrieval
- ⚡ **Real-time Processing:** Live news integration and instant predictions  
- 📊 **Extended Validation:** Multi-month backtesting and robustness analysis
- 🎯 **Multi-asset Support:** Expanding beyond SOL to other cryptocurrencies

---