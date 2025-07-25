#!/usr/bin/env python3
"""
05_predict_with_memory.py

Inference-time memory system for SOL price prediction using:
- STM (Short-Term Memory): Rolling 3 weeks of recent predictions and outcomes
- LTM (Long-Term Memory): Similar market conditions from training weeks 1-40

Process:
1. Load test weeks 41-51 
2. For each week, generate both baseline (no memory) and memory-enhanced predictions
3. STM: Include last 3 weeks of test predictions, outcomes, and learnings
4. LTM: Find 3 most similar training examples using our market feature embedding system
5. Compare performance between memory vs baseline approaches
"""

import pandas as pd
import numpy as np
import pickle
import json
import faiss
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system
except ImportError:
    print("Error: xai_sdk not installed. Please install with: pip install xai-sdk")
    exit(1)

class LTMSystem:
    """Long-Term Memory system using market condition similarity."""
    
    def __init__(self):
        self.index = None
        self.metadata = None
        self.scaler = None
        self.feature_names = None
        self.loaded = False
    
    def load_ltm_system(self) -> bool:
        """Load the LTM components from memory/ directory."""
        try:
            # Load FAISS index
            self.index = faiss.read_index("memory/ltm_faiss_index.bin")
            
            # Load metadata
            with open("memory/ltm_metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
            
            # Load scaler
            with open("memory/ltm_scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            
            # Load feature names
            with open("memory/ltm_feature_names.json", "r") as f:
                self.feature_names = json.load(f)
            
            print(f"✅ Loaded LTM system: {len(self.metadata)} training examples")
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load LTM system: {e}")
            self.loaded = False
            return False
    
    def extract_week_features(self, week_data: pd.Series) -> np.ndarray:
        """Extract market features for a test week (same as training)."""
        try:
            # Core 30-day changes
            sol_change = week_data['sol_30d_change_pct']
            btc_change = week_data['btc_30d_change_pct'] 
            eth_change = week_data['eth_30d_change_pct']
            sp_change = week_data['sp500_30d_change_pct']
            
            # Normalized SOL price level (using training set range)
            sol_price = week_data['prediction_date_sol_price']
            sol_price_norm = (sol_price - 100) / (200 - 100)  # Approximate normalization
            
            # Market volatility indicator
            changes = [abs(sol_change), abs(btc_change), abs(eth_change), abs(sp_change)]
            market_volatility = np.mean(changes)
            
            # Cross-asset correlation pattern
            crypto_changes = [sol_change, btc_change, eth_change]
            correlation_strength = np.std(crypto_changes)
            
            # Tweet sentiment score
            tweet_data = str(week_data.get('tweets', ''))
            tweet_sentiment = min(len(tweet_data) / 2000.0, 1.0)
            
            # Market stress indicator
            stress_changes = [min(c, 0) for c in [sol_change, btc_change, eth_change]]
            market_stress = abs(np.mean(stress_changes))
            
            # Trend strength
            trend_directions = [1 if c > 0 else -1 for c in [sol_change, btc_change, eth_change]]
            trend_strength = abs(np.mean(trend_directions))
            
            # Combine into feature vector
            features = np.array([
                sol_change, btc_change, eth_change, sp_change,
                sol_price_norm, market_volatility, correlation_strength,
                tweet_sentiment, market_stress, trend_strength
            ], dtype=np.float32).reshape(1, -1)
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def find_similar_examples(self, week_data: pd.Series, k: int = 3) -> List[Dict[str, Any]]:
        """Find k most similar training examples based on market conditions."""
        if not self.loaded:
            return []
        
        try:
            # Extract features for current week
            features = self.extract_week_features(week_data)
            if features is None:
                return []
            
            # Normalize features using training scaler
            features_normalized = self.scaler.transform(features).astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(features_normalized)
            
            # Search for similar examples
            scores, indices = self.index.search(features_normalized, k)
            
            similar_examples = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    example = self.metadata[idx].copy()
                    example['similarity_score'] = float(score)
                    similar_examples.append(example)
            
            return similar_examples
            
        except Exception as e:
            print(f"❌ Error finding similar examples: {e}")
            return []

class STMSystem:
    """Short-Term Memory system for rolling test predictions."""
    
    def __init__(self):
        self.test_predictions = []
    
    def add_prediction(self, prediction_data: Dict[str, Any]):
        """Add a completed test prediction to STM."""
        self.test_predictions.append(prediction_data)
    
    def get_stm_context(self, max_weeks: int = 3) -> str:
        """Get STM context from last N test weeks."""
        if not self.test_predictions:
            return ""
        
        # Get the last few predictions
        recent = self.test_predictions[-max_weeks:]
        
        stm_context = "🧠 SHORT-TERM MEMORY (Last 3 test weeks):\n"
        for i, pred in enumerate(recent, 1):
            direction = "✅ CORRECT" if pred['direction_correct'] else "❌ WRONG"
            stm_context += f"Week {pred['week_number']}: Predicted {pred['predicted_change_pct']:+.1f}%, "
            stm_context += f"Actual {pred['actual_change_pct']:+.1f}% ({direction})\n"
            stm_context += f"  Reasoning: {pred['reasoning'][:100]}...\n"
            if pred.get('reflection'):
                stm_context += f"  Learning: {pred['reflection'][:100]}...\n"
            stm_context += "\n"
        
        return stm_context

class MemoryPredictionSystem:
    """Main system for memory-enhanced predictions."""
    
    def __init__(self):
        self.ltm = LTMSystem()
        self.stm = STMSystem()
        self.grok_client = None
        
    def initialize(self) -> bool:
        """Initialize the prediction system."""
        # Load LTM system
        if not self.ltm.load_ltm_system():
            print("⚠️  LTM system failed to load - memory features disabled")
        
        # Initialize Grok client
        api_key = os.getenv('GROK_API_KEY')
        if not api_key:
            print("❌ GROK_API_KEY not found in environment")
            return False
            
        self.grok_client = Client(api_key=api_key)
        print("✅ Grok client initialized")
        
        return True
    
    def generate_baseline_prediction(self, week_data: pd.Series) -> Tuple[Optional[float], Optional[str]]:
        """Generate baseline prediction without memory."""
        
        target_date = week_data['target_date']
        prediction_date = week_data['prediction_date']
        current_price = week_data['prediction_date_sol_price']
        tweets = str(week_data.get('tweets', ''))[:1000]
        
        # Market context
        market_context = f"""
BTC 30-day change: {week_data['btc_30d_change_pct']:.1f}%
ETH 30-day change: {week_data['eth_30d_change_pct']:.1f}%
SOL 30-day change: {week_data['sol_30d_change_pct']:.1f}%
S&P 500 30-day change: {week_data['sp500_30d_change_pct']:.1f}%

Recent tweets: {tweets[:500]}...
        """.strip()
        
        prompt = f"""You are a crypto analyst predicting Solana (SOL) price.

CURRENT SITUATION (as of {target_date}):
- Current SOL Price: ${current_price:.2f}
- Prediction for: {prediction_date} (7 days ahead)

MARKET CONTEXT:
{market_context}

RULES:
- STRICT NO LOOKAHEAD BIAS: Only use data available on {target_date}
- Consider technical trends, market sentiment, fundamentals
- SOL can move 10-30% weekly due to crypto volatility

Provide your prediction in this EXACT format:

PREDICTED_PRICE: [number only, e.g., 162.50]
REASONING: [2-3 sentences explaining your prediction based on available data]

Make a specific price prediction for {prediction_date}."""

        try:
            # Create chat with Grok
            chat = self.grok_client.chat.create(
                model="grok-4-0709",
                temperature=0.2
            )
            
            # Add system message
            chat.append(system("You are a professional cryptocurrency analyst specializing in SOL price prediction."))
            
            # Add user prompt
            chat.append(user(prompt))
            
            # Get response
            response = chat.sample()
            
            return self._parse_prediction(response.content.strip())
            
        except Exception as e:
            print(f"❌ Error in baseline prediction: {e}")
            return None, None
    
    def generate_memory_prediction(self, week_data: pd.Series, week_number: int) -> Tuple[Optional[float], Optional[str]]:
        """Generate memory-enhanced prediction with STM and LTM."""
        
        target_date = week_data['target_date']
        prediction_date = week_data['prediction_date']
        current_price = week_data['prediction_date_sol_price']
        tweets = str(week_data.get('tweets', ''))[:1000]
        
        # Get STM context (last 3 test weeks)
        stm_context = self.stm.get_stm_context()
        
        # Get LTM context (similar training examples)
        ltm_examples = self.ltm.find_similar_examples(week_data)
        ltm_context = self._format_ltm_context(ltm_examples)
        
        # Market context
        market_context = f"""
BTC 30-day change: {week_data['btc_30d_change_pct']:.1f}%
ETH 30-day change: {week_data['eth_30d_change_pct']:.1f}%
SOL 30-day change: {week_data['sol_30d_change_pct']:.1f}%
S&P 500 30-day change: {week_data['sp500_30d_change_pct']:.1f}%

Recent tweets: {tweets[:400]}...
        """.strip()
        
        # Build memory-enhanced prompt
        memory_context = ""
        if stm_context:
            memory_context += stm_context + "\n"
        if ltm_context:
            memory_context += ltm_context + "\n"
        
        prompt = f"""You are an expert crypto analyst with access to historical performance data and recent prediction outcomes.

{memory_context}

CURRENT PREDICTION TASK (as of {target_date}):
- Current SOL Price: ${current_price:.2f}
- Prediction for: {prediction_date} (7 days ahead)

MARKET CONTEXT:
{market_context}

INSTRUCTIONS:
- Learn from recent test performance (STM) and similar historical situations (LTM)
- Consider what prediction approaches worked vs failed in similar conditions
- Factor in technical trends, market sentiment, fundamentals
- STRICT NO LOOKAHEAD BIAS: Only use data available on {target_date}
- SOL can move 10-30% weekly due to crypto volatility

Provide your prediction in this EXACT format:

PREDICTED_PRICE: [number only, e.g., 162.50]
REASONING: [2-3 sentences incorporating lessons from memory and current analysis]

Make a specific price prediction for {prediction_date}."""

        try:
            # Create chat with Grok
            chat = self.grok_client.chat.create(
                model="grok-4-0709",
                temperature=0.2
            )
            
            # Add system message
            chat.append(system("You are a professional cryptocurrency analyst with access to historical performance data and recent outcomes."))
            
            # Add user prompt
            chat.append(user(prompt))
            
            # Get response
            response = chat.sample()
            
            return self._parse_prediction(response.content.strip())
            
        except Exception as e:
            print(f"❌ Error in memory prediction: {e}")
            return None, None
    
    def _format_ltm_context(self, examples: List[Dict[str, Any]]) -> str:
        """Format LTM examples for the prompt."""
        if not examples:
            return ""
        
        ltm_context = "🔍 LONG-TERM MEMORY (Similar historical conditions):\n"
        
        for i, ex in enumerate(examples, 1):
            direction = "✅" if ex['direction_correct'] else "❌"
            ltm_context += f"Week {ex['week_index']} (similarity {ex['similarity_score']:.2f}): "
            ltm_context += f"Predicted {ex['predicted_change_pct']:+.1f}%, Actual {ex['actual_change_pct']:+.1f}% ({direction})\n"
            ltm_context += f"  Market: SOL {ex['market_features']['sol_30d_change']:.1f}%, BTC {ex['market_features']['btc_30d_change']:.1f}%\n"
            ltm_context += f"  Reasoning: {ex['llm_reasoning'][:80]}...\n"
            ltm_context += f"  Learning: {ex['llm_reflection'][:80]}...\n\n"
        
        return ltm_context
    
    def _parse_prediction(self, content: str) -> Tuple[Optional[float], Optional[str]]:
        """Parse Grok response to extract price and reasoning."""
        import re
        
        predicted_price = None
        reasoning = ""
        
        # Extract price
        price_match = re.search(r'PREDICTED_PRICE:\s*[\$]?(\d+\.?\d*)', content, re.IGNORECASE)
        if price_match:
            try:
                predicted_price = float(price_match.group(1))
            except ValueError:
                pass
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            reasoning = content.strip()
        
        # Fallback price extraction
        if predicted_price is None:
            numbers = re.findall(r'\d+\.?\d*', content)
            for num in numbers:
                try:
                    price = float(num)
                    if 50 < price < 500:  # Reasonable SOL price range
                        predicted_price = price
                        break
                except ValueError:
                    continue
        
        return predicted_price, reasoning
    
    def generate_reflection(self, week_data: pd.Series, predicted_price: float, predicted_change: float, 
                          reasoning: str, is_memory: bool) -> Optional[str]:
        """Generate reflection on prediction accuracy."""
        
        target_date = week_data['target_date']
        current_price = week_data['prediction_date_sol_price']
        actual_price = week_data['target_date_sol_price']
        actual_change = week_data['actual_change_pct']
        
        direction_correct = (predicted_change > 0) == (actual_change > 0)
        method = "memory-enhanced" if is_memory else "baseline"
        
        prompt = f"""Analyze this {method} SOL prediction result:

PREDICTION SUMMARY ({target_date}):
- Method: {method.upper()}
- Start Price: ${current_price:.2f}
- Predicted: ${predicted_price:.2f} ({predicted_change:+.1f}%)
- Actual: ${actual_price:.2f} ({actual_change:+.1f}%)
- Direction: {'CORRECT' if direction_correct else 'WRONG'}
- Price Error: ${abs(predicted_price - actual_price):.2f}

ORIGINAL REASONING: {reasoning}

Provide a 2-3 sentence reflection on why this {method} prediction succeeded or failed. Focus on what can be learned for future predictions. Be specific about market factors or reasoning quality."""

        try:
            # Create chat with Grok
            chat = self.grok_client.chat.create(
                model="grok-4-0709",
                temperature=0.2
            )
            
            # Add system message
            chat.append(system("You analyze prediction performance to extract actionable insights for improvement."))
            
            # Add user prompt
            chat.append(user(prompt))
            
            # Get response
            response = chat.sample()
            
            return response.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating reflection: {e}")
            return None

def main():
    """Main function to run memory vs baseline comparison."""
    
    print("🧠 INFERENCE-TIME MEMORY COMPARISON")
    print("=" * 80)
    print("Testing STM (Short-Term Memory) + LTM (Long-Term Memory) vs Baseline")
    print("Test Set: Weeks 41-51 | Training: Weeks 1-40")
    print("=" * 80)
    
    # Initialize system
    system = MemoryPredictionSystem()
    if not system.initialize():
        print("❌ System initialization failed")
        return
    
    # Load test data
    try:
        df = pd.read_csv("test_set_results.csv")
        print(f"✅ Loaded {len(df)} test weeks")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    successful_baseline = 0
    successful_memory = 0
    
    # Process each test week
    for i, (idx, row) in enumerate(df.iterrows()):
        week_number = i + 41  # Weeks 41-51
        
        print(f"\n{'='*60}")
        print(f"📅 WEEK {week_number} ({row['target_date']})")
        print(f"💰 Current SOL: ${row['prediction_date_sol_price']:.2f}")
        print(f"🎯 Actual SOL: ${row['target_date_sol_price']:.2f} ({row['actual_change_pct']:+.1f}%)")
        print(f"📈 Market: BTC {row['btc_30d_change_pct']:+.1f}%, ETH {row['eth_30d_change_pct']:+.1f}%, SOL {row['sol_30d_change_pct']:+.1f}%")
        print(f"{'='*60}")
        
        # 1. BASELINE PREDICTION
        print("\n🤖 Generating baseline prediction (no memory)...")
        baseline_price, baseline_reasoning = system.generate_baseline_prediction(row)
        
        if baseline_price is not None:
            baseline_change = ((baseline_price - row['prediction_date_sol_price']) / row['prediction_date_sol_price']) * 100
            baseline_direction = (baseline_change > 0) == (row['actual_change_pct'] > 0)
            
            print(f"📊 Baseline: ${baseline_price:.2f} ({baseline_change:+.1f}%) - {'✅' if baseline_direction else '❌'}")
            
            # Generate baseline reflection
            baseline_reflection = system.generate_reflection(row, baseline_price, baseline_change, baseline_reasoning, False)
            
            # Save baseline results
            df.at[idx, 'predicted_price_baseline'] = baseline_price
            df.at[idx, 'predicted_change_pct_baseline'] = baseline_change
            df.at[idx, 'llm_reasoning_baseline'] = baseline_reasoning
            df.at[idx, 'llm_reflection_baseline'] = baseline_reflection or "Reflection failed"
            
            successful_baseline += 1
        else:
            print("❌ Baseline prediction failed")
        
        # 2. MEMORY-ENHANCED PREDICTION
        print("\n🧠 Generating memory-enhanced prediction...")
        memory_price, memory_reasoning = system.generate_memory_prediction(row, week_number)
        
        if memory_price is not None:
            memory_change = ((memory_price - row['prediction_date_sol_price']) / row['prediction_date_sol_price']) * 100
            memory_direction = (memory_change > 0) == (row['actual_change_pct'] > 0)
            
            print(f"📊 Memory: ${memory_price:.2f} ({memory_change:+.1f}%) - {'✅' if memory_direction else '❌'}")
            
            # Generate memory reflection
            memory_reflection = system.generate_reflection(row, memory_price, memory_change, memory_reasoning, True)
            
            # Save memory results
            df.at[idx, 'predicted_price_memory'] = memory_price
            df.at[idx, 'predicted_change_pct_memory'] = memory_change
            df.at[idx, 'llm_reasoning_memory'] = memory_reasoning
            df.at[idx, 'llm_reflection_memory'] = memory_reflection or "Reflection failed"
            
            # Add to STM for next weeks
            stm_data = {
                'week_number': week_number,
                'predicted_change_pct': memory_change,
                'actual_change_pct': row['actual_change_pct'],
                'direction_correct': memory_direction,
                'reasoning': memory_reasoning,
                'reflection': memory_reflection
            }
            system.stm.add_prediction(stm_data)
            
            successful_memory += 1
        else:
            print("❌ Memory prediction failed")
        
        # Save progress
        df.to_csv("test_set_results.csv", index=False)
        print(f"💾 Progress saved - Week {week_number} completed")
        
        # Brief pause
        time.sleep(3)
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎉 MEMORY COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"📊 Baseline predictions: {successful_baseline}/{len(df)}")
    print(f"🧠 Memory predictions: {successful_memory}/{len(df)}")
    print(f"📄 Results saved to: test_set_results.csv")
    print("\n🎯 Next Steps:")
    print("- Run analyze_test_performance.py to compare memory vs baseline")
    print("- Analyze directional accuracy, MAE, trading performance")
    print("- Evaluate STM and LTM effectiveness")

if __name__ == "__main__":
    main() 