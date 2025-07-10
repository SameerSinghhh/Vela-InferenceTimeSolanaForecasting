# 🧠 Inference-Time LLM Forecasting for Solana (SOL)

## Project Overview

This project builds a novel inference-time LLM forecasting tool that predicts the future price of Solana (SOL) using only inference-time memory (i.e., no training of model weights). The goal is to incrementally "teach" the LLM to reason better by feeding it summaries of previous examples — including what it predicted, what actually happened, and a reflection on why it was wrong — as context at test time.

## Key Innovation

All learning happens at **inference time** through two forms of memory:
- **Short-term memory (STM):** A few most relevant prior examples retrieved based on similarity to the current test case
- **Long-term memory (LTM):** A stored database of all prior training examples (predictions, actual values, summaries, reflections, etc.)

## Setup

### Environment Variables
Create a `.env` file in the project root with your API keys:

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
SERP_API_KEY=your_serp_api_key_here
SERP_BASE_URL=https://csearch.vela.partners/search
EOF
```

### Dependencies
```bash
pip install requests python-dotenv openai beautifulsoup4 lxml newspaper3k requests-html selenium webdriver-manager fake-useragent
```

**⚠️ Security Note:** Never commit API keys to git! The `.env` file is already in `.gitignore`.

## Technical Constraints & APIs

### LLM Model
- **Always use:** `o3-mini` from OpenAI
- **OpenAI API Key:** Set via `OPENAI_API_KEY` environment variable (see Setup section)

### Search API
- **SERP API Endpoint:** `https://csearch.vela.partners/search`
- **SERP API Key:** Set via `SERP_API_KEY` environment variable (see Setup section)

### Price Data Source
- **SOL Price History:** `https://onchainbe.vela.partners/market/history/So11111111111111111111111111111111111111111?days=365`
- **Format:** JSON with `date` (YYYYMMDD) and `price` fields
- **Available Range:** Approximately 1 year of historical data
- **Earliest Training Date:** July 30, 2024

## Project Structure

```
inference_time_model/
│
├── 01_collect_data.py             # ✅ COMPLETE: Price data + CSV structure
├── 02_collect_news.py             # 🎯 CURRENT: Incremental news collection + summarization
├── 03_generate_predictions.py     # LLM predictions + reasoning
├── 04_generate_reflections.py     # LLM reflections on prediction errors
├── 05_embed_training_set.py       # Embeds examples for similarity retrieval (LTM)
├── 06_predict_with_memory.py      # Uses STM + LTM for inference-time predictions
├── 07_evaluate_predictions.py     # Computes metrics on test set performance
│
├── /data
│   ├── raw_news/                  # Raw HTML or JSON files of scraped articles
│   ├── article_tracking.json     # Tracks processed articles to avoid duplicates
│   └── training_set.csv           # 🎯 Live document: progressively enhanced
│
├── /memory
│   ├── faiss_index.bin           # Serialized vector DB index (LTM storage)
│   └── embeddings.pkl            # Metadata tied to each embedded training sample
│
├── /utils
│   ├── news_scraper.py           # Smart article collection and validation
│   ├── openai_llm.py             # Handles prompt formatting + calls to LLM API
│   └── memory_retrieval.py       # Retrieves most similar examples from LTM
│
└── config.py                     # Central config file for parameters
```

## Current Phase: Foundation Complete! ✅

### PHASE 1 COMPLETE: Price Data Collection (`01_collect_data.py`)

Successfully implemented:
1. **✅ Biweekly date generation** from July 17, 2024 to April 9, 2025 (20 training dates)
2. **✅ Complete price data collection** from SOL API with 100% coverage
3. **✅ Accurate change calculations** between target and prediction dates
4. **✅ Live CSV structure** ready for progressive filling
5. **✅ Train/test split** - Training until April 15, 2025, test set after

### Current Status: 
- **Phase 1**: Price foundation ✅
- **Phase 2**: Incremental news collection 🎯 
- **Live CSV**: Ready for progressive enhancement ✅

### Data Collection Strategy
- **Target Schedule:** Biweekly intervals (every 2 weeks) to avoid overlap
- **News Window:** 7 days prior to target date
- **Prediction Window:** 1 week after target date
- **Article Validation:** Must mention "Solana" and be within date range
- **Summary Format:** `[Title]: [1-2 sentence summary focusing on Solana price events]`

### Required Class Structure
```python
class SolanaDataCollector:
    def __init__(self):
        # SERP API setup
        # OpenAI o3-mini client setup
    
    def search_articles(target_date):
        # Uses SERP to retrieve relevant articles
    
    def scrape_and_validate(article_url):
        # Returns raw HTML + basic Solana mention check + date check
    
    def summarize_articles(article_texts):
        # Uses o3-mini to summarize
    
    def collect_and_save_all(target_dates: List[str]):
        # For each date, collect and write row to training_set.csv
```

### Current CSV Schema (`training_set.csv`) ✅
| Column | Description | Status |
|--------|-------------|--------|
| `target_date` | Date of prediction point | ✅ **20 training dates** |
| `context_start` | Start of news context window (target-7 days) | ✅ **Calculated** |
| `target_price` | SOL price on target_date | ✅ **$118-$256 range** |
| `prediction_end` | End of prediction window (target+7 days) | ✅ **Calculated** |
| `actual_price` | SOL price on prediction_end | ✅ **100% coverage** |
| `actual_change_pct` | Actual % change (target → prediction) | ✅ **-15.9% to +24.5%** |
| `summarized_context` | News summaries | ⏳ **Phase 2** |
| `predicted_price` | LLM's price prediction | ⏳ **Phase 3** |
| `predicted_change_pct` | LLM's percentage prediction | ⏳ **Phase 3** |
| `llm_reasoning` | LLM's reasoning | ⏳ **Phase 3** |
| `llm_reflection` | LLM's error reflection | ⏳ **Phase 4** |

## Complete Pipeline Overview

### ✅ Phase 1: Foundation (COMPLETE)
- **Script:** `01_collect_data.py`
- **Output:** `training_set.csv` with price data and structure
- **Achievement:** 20 biweekly training dates with 100% price coverage

### 🎯 Phase 2: Incremental News Collection (CURRENT)
- **Script:** `02_collect_news.py`
- **Strategy:** Smart article reuse + incremental summary building
- **Process:**
  1. Read current CSV state (check existing summaries)
  2. Pull articles for target date ranges
  3. Validate dates + Solana relevance using LLM
  4. Reassign articles to best-fit weeks (cost-efficient reuse)
  5. Merge new articles with existing summaries
  6. Update `summarized_context` column progressively

### Phase 3: Generate Predictions
- **Script:** `03_generate_predictions.py`
- **Process:** Use o3-mini to analyze summaries + make price predictions
- **Columns Added:** `predicted_price`, `predicted_change_pct`, `llm_reasoning`

### Phase 4: Generate Reflections
- **Script:** `04_generate_reflections.py`
- **Process:** Compare predictions vs actual, generate error reflections
- **Columns Added:** `llm_reflection`

### Phase 5: Embed Training Examples (LTM)
- **Script:** `05_embed_training_set.py`
- **Output:** FAISS vector database for similarity retrieval
- **Storage:** `memory/faiss_index.bin` + `memory/embeddings.pkl`

### Phase 6: Inference-Time Prediction
- **Script:** `06_predict_with_memory.py`
- **Memory Integration:**
  - **STM:** Last 3 training examples
  - **LTM:** Top 3 most similar examples (via FAISS)
- **Prompt Structure:**
```
Recent Mistakes (STM):
- Week of 2024-12-03: Predicted +4.2%, Actual +0.5%, Error -3.7%
  Reflection: Overestimated impact of Solana ETF rumors

Relevant Past Situations (LTM):
- Week of 2024-04-01: News: "surge in NFT interest..."
  Outcome: Predicted +3.0%, Actual -2.0%

This Week's News:
- "Solana integrated into Shopify plugin..."

Task: Predict % change in SOL price for next week.
```

### Phase 7: Evaluation
- **Script:** `07_evaluate_predictions.py`
- **Metrics:** MAE, Directional Accuracy, Brier Score
- **Baselines:** No-memory LLM, Classical time-series models

## Data Splits ✅
- **Training Set:** July 17, 2024 – April 9, 2025 (20 biweekly dates)
- **Test Set:** April 15, 2025 onwards (to be generated later)
- **Validation:** Last few weeks of training period (March-April 2025)

## Article Summary Format
Each article should be summarized as:
```
[Article Title]: [1-2 sentence summary focusing on events relevant to Solana price]
```

At the end of all summaries, include a synthesized recap generated by o3-mini that could help predict SOL's future price based on that week's news.

## Key Features
- ✅ **No Model Training:** All learning happens at inference time
- ✅ **Memory-Enhanced:** STM + LTM architecture mimics cognitive reasoning
- ✅ **Self-Reflective:** LLM reflects on past errors to improve future predictions
- ✅ **Biweekly Predictions:** Avoids data overlap and provides realistic prediction intervals
- ✅ **Comprehensive Validation:** Articles must pass relevance and timestamp checks

## Implementation Notes
- Use `datetime` for proper 7-day window calculations
- Ensure CSV can be appended to (don't overwrite each time)
- Add basic logging for visibility
- Validate articles contain "Solana" mentions
- Verify article publication dates fall within target range

## Research Impact
This project is novel because:
1. **Inference-time learning** without weight updates
2. **Memory architecture** for time-series prediction
3. **Self-reflection mechanisms** for error correction
4. **Cognitive reasoning simulation** in financial forecasting

**Publishable Angle:** Memory-enhanced inference-time LLMs for time-series prediction in volatile financial markets. 