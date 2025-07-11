import os
import pandas as pd
import openai
import faiss
import pickle
import numpy as np
from datetime import datetime
import time
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class MemorySystem:
    """Handles loading and querying the FAISS-based memory system."""
    
    def __init__(self):
        self.index = None
        self.metadata = None
        self.loaded = False
    
    def load_memory(self) -> bool:
        """Load the FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index("memory/faiss_index.bin")
            
            # Load metadata
            with open("memory/embeddings.pkl", "rb") as f:
                self.metadata = pickle.load(f)
            
            print(f"✅ Loaded memory system: {len(self.metadata)} training examples")
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load memory system: {e}")
            self.loaded = False
            return False
    
    def find_similar_examples(self, query_context: str, k: int = 3) -> List[Dict[str, Any]]:
        """Find k most similar training examples to the query context."""
        if not self.loaded:
            return []
        
        try:
            # Generate embedding for query
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[query_context]
            )
            
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search for similar examples
            scores, indices = self.index.search(query_embedding, k)
            
            similar_examples = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):  # Verify valid index
                    example = self.metadata[idx].copy()
                    example['similarity_score'] = float(score)
                    similar_examples.append(example)
            
            return similar_examples
            
        except Exception as e:
            print(f"❌ Error finding similar examples: {e}")
            return []

class TestPredictionSystem:
    """Handles progressive test predictions with STM and LTM."""
    
    def __init__(self):
        self.memory_system = MemorySystem()
        self.test_predictions = []  # Store predictions as we make them for STM
    
    def load_memory(self) -> bool:
        """Load the memory system."""
        return self.memory_system.load_memory()
    
    def add_test_prediction(self, prediction_data: Dict[str, Any]):
        """Add a completed test prediction to STM."""
        self.test_predictions.append(prediction_data)
    
    def get_stm_context(self, current_week: int, max_weeks: int = 2) -> str:
        """Get Short-term Memory context from recent test predictions."""
        if not self.test_predictions:
            return ""
        
        # Get the last few test predictions
        recent_predictions = self.test_predictions[-max_weeks:]
        
        stm_context = "RECENT PERFORMANCE (Short-term Memory):\n"
        for pred in recent_predictions:
            direction = "✅ CORRECT" if pred['direction_correct'] else "❌ WRONG"
            stm_context += f"- Week {pred['target_date']}: Predicted {pred['predicted_change_pct']:+.1f}%, Actual {pred['actual_change_pct']:+.1f}% ({direction})\n"
            if pred.get('llm_reflection'):
                stm_context += f"  Reflection: {pred['llm_reflection'][:100]}...\n"
        
        return stm_context + "\n"
    
    def get_ltm_context(self, query_context: str, k: int = 3) -> str:
        """Get Long-term Memory context from similar training examples."""
        similar_examples = self.memory_system.find_similar_examples(query_context, k)
        
        if not similar_examples:
            return ""
        
        ltm_context = "SIMILAR PAST SITUATIONS (Long-term Memory):\n"
        for i, example in enumerate(similar_examples, 1):
            direction = "✅" if (example['predicted_change_pct'] > 0 and example['actual_change_pct'] > 0) or (example['predicted_change_pct'] < 0 and example['actual_change_pct'] < 0) else "❌"
            ltm_context += f"{i}. Week {example['target_date']} (similarity: {example['similarity_score']:.3f})\n"
            ltm_context += f"   Context: {example['summarized_context'][:120]}...\n"
            ltm_context += f"   Predicted: {example['predicted_change_pct']:+.1f}%, Actual: {example['actual_change_pct']:+.1f}% {direction}\n"
            if example.get('llm_reflection'):
                ltm_context += f"   Lesson: {example['llm_reflection'][:100]}...\n"
            ltm_context += "\n"
        
        return ltm_context

def generate_baseline_prediction(target_date: str, current_price: float, context: str, prediction_end_date: str) -> Tuple[Optional[float], Optional[str]]:
    """Generate baseline prediction without memory (same as Phase 3)."""
    
    target_date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%B %d, %Y')
    prediction_end_str = datetime.strptime(prediction_end_date, '%Y-%m-%d').strftime('%B %d, %Y')
    
    short_context = context[:300] + "..." if len(context) > 300 else context
    
    prediction_prompt = f"""You are a crypto analyst predicting Solana (SOL) price.

CURRENT DATA:
- Date: {target_date_str}
- Current SOL Price: ${current_price:.2f}
- Target Date: {prediction_end_str} (7 days ahead)

CONTEXT:
{short_context}

RULES:
- Only use information available on or before {target_date_str}
- Consider technical trends, market sentiment, and fundamentals
- SOL can move 10-30% in a week due to crypto volatility

Provide your prediction in this exact format:

PREDICTED_PRICE: [number only, e.g., 162.50]
REASONING: [2-3 sentences explaining your prediction]

Make a specific price prediction for 7 days from now."""

    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a professional cryptocurrency analyst with expertise in Solana (SOL) price prediction. You provide clear, data-driven predictions with concise reasoning."},
                {"role": "user", "content": prediction_prompt}
            ],
            max_completion_tokens=1000
        )
        
        return parse_prediction_response(response.choices[0].message.content.strip())
        
    except Exception as e:
        print(f"❌ Error in baseline prediction: {e}")
        return None, None

def generate_memory_enhanced_prediction(target_date: str, current_price: float, context: str, prediction_end_date: str, 
                                      stm_context: str, ltm_context: str) -> Tuple[Optional[float], Optional[str]]:
    """Generate memory-enhanced prediction with STM and LTM context."""
    
    target_date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%B %d, %Y')
    prediction_end_str = datetime.strptime(prediction_end_date, '%Y-%m-%d').strftime('%B %d, %Y')
    
    short_context = context[:200] + "..." if len(context) > 200 else context
    
    # Build enhanced prompt with memory context
    memory_context = ""
    if stm_context:
        memory_context += stm_context
    if ltm_context:
        memory_context += ltm_context
    
    prediction_prompt = f"""You are an expert crypto analyst predicting Solana (SOL) price with access to historical performance data.

{memory_context}

CURRENT PREDICTION TASK:
- Date: {target_date_str}
- Current SOL Price: ${current_price:.2f}
- Target Date: {prediction_end_str} (7 days ahead)

THIS WEEK'S CONTEXT:
{short_context}

INSTRUCTIONS:
- Learn from the recent performance and similar past situations shown above
- Consider what worked and what didn't in similar market conditions
- Factor in technical trends, market sentiment, and fundamentals
- SOL can move 10-30% in a week due to crypto volatility
- Only use information available on or before {target_date_str}

Provide your prediction in this exact format:

PREDICTED_PRICE: [number only, e.g., 162.50]
REASONING: [2-3 sentences explaining your prediction, incorporating lessons from memory]

Make a specific price prediction for 7 days from now."""

    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a professional cryptocurrency analyst with access to historical performance data. You learn from past successes and failures to improve your predictions."},
                {"role": "user", "content": prediction_prompt}
            ],
            max_completion_tokens=1500
        )
        
        return parse_prediction_response(response.choices[0].message.content.strip())
        
    except Exception as e:
        print(f"❌ Error in memory-enhanced prediction: {e}")
        return None, None

def parse_prediction_response(content: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse the LLM response to extract price and reasoning."""
    import re
    
    predicted_price = None
    reasoning = ""
    
    # Extract price
    price_patterns = [
        r'PREDICTED_PRICE:\s*[\$]?(\d+\.?\d*)',
        r'price.*?[\$]?(\d+\.?\d*)',
        r'target.*?[\$]?(\d+\.?\d*)',
        r'predict.*?[\$]?(\d+\.?\d*)',
        r'[\$](\d+\.?\d*)'
    ]
    
    for pattern in price_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            try:
                predicted_price = float(matches[0])
                break
            except ValueError:
                continue
    
    # Extract reasoning
    reasoning_patterns = [
        r'REASONING:\s*(.+)',
        r'reasoning.*?:\s*(.+)',
        r'because\s+(.+)',
        r'expect.*?(\w.+)',
        r'analysis.*?(\w.+)'
    ]
    
    for pattern in reasoning_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        if matches:
            reasoning = matches[0].strip()
            break
    
    if not reasoning:
        reasoning = content.strip()
    
    if predicted_price is None:
        # Try to extract any reasonable price
        all_numbers = re.findall(r'\d+\.?\d*', content)
        if all_numbers:
            numbers = [float(n) for n in all_numbers if 50 < float(n) < 500]
            if numbers:
                predicted_price = numbers[0]
    
    return predicted_price, reasoning

def generate_reflection(target_date: str, current_price: float, predicted_price: float, actual_price: float, 
                       context: str, reasoning: str, is_memory_enhanced: bool = False) -> Optional[str]:
    """Generate reflection on prediction accuracy."""
    
    predicted_change = ((predicted_price - current_price) / current_price) * 100
    actual_change = ((actual_price - current_price) / current_price) * 100
    
    target_date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%B %d, %Y')
    short_context = context[:200] + "..." if len(context) > 200 else context
    
    method_type = "memory-enhanced" if is_memory_enhanced else "baseline"
    
    reflection_prompt = f"""Analyze this {method_type} SOL price prediction:

PREDICTION RESULTS:
- Date: {target_date_str}
- Start: ${current_price:.2f}
- Predicted: ${predicted_price:.2f} ({predicted_change:.1f}%)
- Actual: ${actual_price:.2f} ({actual_change:.1f}%)
- Direction: {"CORRECT" if (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0) else "WRONG"}

CONTEXT: {short_context}

REASONING: {reasoning}

Explain in 2-3 sentences why this {method_type} prediction was right or wrong. Focus on market factors, reasoning quality, and what could be learned for future predictions."""

    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are analyzing prediction performance to identify lessons for future forecasting."},
                {"role": "user", "content": reflection_prompt}
            ],
            max_completion_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ Error generating reflection: {e}")
        return None

def process_test_set(file_path: str, use_memory: bool = False, force_reprocess: bool = True) -> int:
    """Process a test set with or without memory enhancement."""
    
    print(f"\n{'='*60}")
    print(f"🚀 Processing: {file_path}")
    print(f"📊 Mode: {'Memory-Enhanced' if use_memory else 'Baseline (No Memory)'}")
    print(f"🔄 Reprocess: {'Yes (overwriting existing)' if force_reprocess else 'No (skip completed)'}")
    print(f"{'='*60}")
    
    # Load test data
    df = pd.read_csv(file_path)
    total_weeks = len(df)
    
    # Initialize memory system if needed
    prediction_system = None
    if use_memory:
        prediction_system = TestPredictionSystem()
        if not prediction_system.load_memory():
            print("❌ Failed to load memory system, falling back to baseline mode")
            use_memory = False
    
    successful_predictions = 0
    
    for i, row in df.iterrows():
        # Check if already processed (only skip if force_reprocess is False)
        already_processed = pd.notna(row['predicted_price']) and pd.notna(row['llm_reasoning'])
        if already_processed and not force_reprocess:
            print(f"Week {i+1}/{total_weeks} - {row['target_date']}: Already processed, skipping...")
            successful_predictions += 1  # Count as successful since it exists
            continue
        elif already_processed and force_reprocess:
            print(f"Week {i+1}/{total_weeks} - {row['target_date']}: Reprocessing (overwriting existing)...")
        
        print(f"\n📅 Week {i+1}/{total_weeks}: {row['target_date']}")
        print(f"💰 Current Price: ${row['target_price']:.2f}")
        print(f"🎯 Actual Price: ${row['actual_price']:.2f} ({row['actual_change_pct']:+.2f}%)")
        print(f"📰 Context: {row['summarized_context'][:100]}...")
        
        # Retry logic for prediction generation
        max_retries = 3
        predicted_price, reasoning = None, None
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"🔄 Retry attempt {attempt + 1}/{max_retries}")
                time.sleep(5)  # Wait longer between retries
            
            # Generate prediction
            if use_memory and prediction_system:
                # Get memory context
                stm_context = prediction_system.get_stm_context(i)
                ltm_context = prediction_system.get_ltm_context(row['summarized_context'])
                
                print(f"🧠 Using STM: {len(stm_context) > 0}")
                print(f"🧠 Using LTM: {len(ltm_context) > 0}")
                
                predicted_price, reasoning = generate_memory_enhanced_prediction(
                    row['target_date'], row['target_price'], row['summarized_context'],
                    row['prediction_end'], stm_context, ltm_context
                )
            else:
                predicted_price, reasoning = generate_baseline_prediction(
                    row['target_date'], row['target_price'], row['summarized_context'],
                    row['prediction_end']
                )
            
            # Check if prediction was successful
            if predicted_price is not None and reasoning is not None:
                break
            else:
                print(f"❌ Attempt {attempt + 1} failed - predicted_price: {predicted_price}, reasoning: {reasoning is not None}")
        
        # If all retries failed, skip this week but report the failure
        if predicted_price is None or reasoning is None:
            print(f"❌ FAILED: Could not generate prediction for week {i+1} after {max_retries} attempts")
            print(f"   This week will be skipped and marked as failed")
            continue
        
        # Calculate prediction metrics
        predicted_change_pct = ((predicted_price - row['target_price']) / row['target_price']) * 100
        direction_correct = (predicted_change_pct > 0 and row['actual_change_pct'] > 0) or \
                           (predicted_change_pct < 0 and row['actual_change_pct'] < 0)
        
        print(f"🤖 Predicted: ${predicted_price:.2f} ({predicted_change_pct:+.2f}%)")
        print(f"📊 Direction: {'✅ CORRECT' if direction_correct else '❌ WRONG'}")
        
        # Generate reflection with retry logic
        reflection = None
        for attempt in range(2):  # Fewer retries for reflection
            if attempt > 0:
                print(f"🔄 Reflection retry {attempt + 1}")
                time.sleep(3)
            
            reflection = generate_reflection(
                row['target_date'], row['target_price'], predicted_price, row['actual_price'],
                row['summarized_context'], reasoning, use_memory
            )
            
            if reflection is not None:
                break
        
        # Update DataFrame
        df.at[i, 'predicted_price'] = predicted_price
        df.at[i, 'predicted_change_pct'] = predicted_change_pct
        df.at[i, 'llm_reasoning'] = str(reasoning)  # Convert to string to avoid dtype warning
        df.at[i, 'llm_reflection'] = str(reflection or "Reflection generation failed")
        
        # Save progress immediately
        df.to_csv(file_path, index=False)
        print(f"💾 Saved progress to {file_path}")
        
        # Add to STM if using memory
        if use_memory and prediction_system:
            test_prediction = {
                'target_date': row['target_date'],
                'predicted_change_pct': predicted_change_pct,
                'actual_change_pct': row['actual_change_pct'],
                'direction_correct': direction_correct,
                'llm_reflection': reflection
            }
            prediction_system.add_test_prediction(test_prediction)
        
        successful_predictions += 1
        
        # Brief pause between predictions
        time.sleep(2)
    
    print(f"\n✅ Completed {successful_predictions}/{total_weeks} predictions for {file_path}")
    if successful_predictions < total_weeks:
        print(f"⚠️  {total_weeks - successful_predictions} predictions failed after all retries")
    
    return successful_predictions

def main():
    """Main function to run Phase 5: Memory-Enhanced Predictions."""
    
    print("🧠 PHASE 5: INFERENCE-TIME MEMORY PREDICTIONS")
    print("=" * 80)
    print("Comparing o3-mini WITH memory vs WITHOUT memory")
    print("STM: Short-term Memory (recent test predictions)")
    print("LTM: Long-term Memory (similar training examples)")
    print("=" * 80)
    
    # Process baseline predictions (no memory)
    print("\n🔄 Step 1: Generating baseline predictions (no memory)...")
    baseline_success = process_test_set("test_set_no_memory.csv", use_memory=False, force_reprocess=True)
    
    # Process memory-enhanced predictions
    print("\n🔄 Step 2: Generating memory-enhanced predictions...")
    memory_success = process_test_set("test_set_with_memory.csv", use_memory=True, force_reprocess=True)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 PHASE 5 COMPLETION SUMMARY:")
    print(f"{'='*60}")
    print(f"Baseline predictions: {baseline_success} completed")
    print(f"Memory-enhanced predictions: {memory_success} completed")
    print("\n🎯 Next Steps:")
    print("- Phase 6: Run 06_evaluate_predictions.py to compare performance")
    print("- Compare directional accuracy, MAE, and other metrics")
    print("- Analyze which memory components (STM vs LTM) help most")

if __name__ == "__main__":
    main() 