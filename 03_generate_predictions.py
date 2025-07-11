import os
import pandas as pd
import openai
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_prediction(target_date, current_price, context, prediction_end_date):
    """
    Generate a price prediction using o3-mini based on current price and context.
    """
    
    # Convert dates to readable format
    target_date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%B %d, %Y')
    prediction_end_str = datetime.strptime(prediction_end_date, '%Y-%m-%d').strftime('%B %d, %Y')
    
    # Truncate context to avoid token limits
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
        
        content = response.choices[0].message.content.strip()
        
        # Parse the response - be more flexible
        lines = content.split('\n')
        predicted_price = None
        reasoning = ""
        
        # Extract price from anywhere in the response
        import re
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
        
        # If no reasoning found, use the whole response
        if not reasoning:
            reasoning = content.strip()
        
        if predicted_price is None:
            # Try to extract any price-like number from the response
            all_numbers = re.findall(r'\d+\.?\d*', content)
            if all_numbers:
                # Find the number closest to the current price
                numbers = [float(n) for n in all_numbers if 50 < float(n) < 500]  # Reasonable SOL price range
                if numbers:
                    predicted_price = numbers[0]
        
        if predicted_price is None:
            raise ValueError(f"Could not parse predicted price from response: {content}")
        
        return predicted_price, reasoning
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

def generate_reflection(target_date, current_price, predicted_price, actual_price, context, reasoning):
    """
    Generate a reflection on why the prediction was right or wrong.
    """
    
    predicted_change = ((predicted_price - current_price) / current_price) * 100
    actual_change = ((actual_price - current_price) / current_price) * 100
    
    target_date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%B %d, %Y')
    
    # Truncate context for reflection
    short_context = context[:200] + "..." if len(context) > 200 else context
    
    reflection_prompt = f"""Analyze this SOL price prediction:

PREDICTION:
- Date: {target_date_str}
- Start: ${current_price:.2f}
- Predicted: ${predicted_price:.2f} ({predicted_change:.1f}%)
- Actual: ${actual_price:.2f} ({actual_change:.1f}%)
- Direction: {"CORRECT" if (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0) else "WRONG"}

CONTEXT: {short_context}

REASONING: {reasoning}

Explain in 2-3 sentences why this prediction was right or wrong. Consider market factors, reasoning quality, and crypto volatility."""

    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a professional analyst providing objective reflections on cryptocurrency predictions. You focus on market factors and analytical reasoning."},
                {"role": "user", "content": reflection_prompt}
            ],
            max_completion_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in reflection: {e}")
        return None

def process_all_weeks():
    """
    Process all weeks of data to generate predictions and reflections.
    """
    
    # Load the training data
    df = pd.read_csv('training_set.csv')
    
    total_weeks = len(df)
    processed_count = 0
    success_count = 0
    
    print(f"Processing {total_weeks} weeks of data...")
    print("=" * 60)
    
    for i, row in df.iterrows():
        # Skip if already processed (has prediction)
        if pd.notna(row['predicted_price']) and pd.notna(row['llm_reasoning']):
            print(f"Week {i+1}/{total_weeks} - {row['target_date']}: Already processed, skipping...")
            continue
        
        print(f"\nProcessing Week {i+1}/{total_weeks}:")
        print(f"Target Date: {row['target_date']}")
        print(f"Current Price: ${row['target_price']:.2f}")
        print(f"Actual Price: ${row['actual_price']:.2f}")
        print(f"Actual Change: {row['actual_change_pct']:.2f}%")
        print(f"Context: {row['summarized_context'][:100]}...")
        print("\n" + "-"*40)
        
        # Generate prediction
        print("Generating prediction...")
        predicted_price, reasoning = generate_prediction(
            row['target_date'],
            row['target_price'],
            row['summarized_context'],
            row['prediction_end']
        )
        
        if predicted_price is None:
            print(f"❌ Failed to generate prediction for week {i+1}")
            continue
        
        # Calculate predicted change percentage
        predicted_change_pct = ((predicted_price - row['target_price']) / row['target_price']) * 100
        
        print(f"✅ Predicted Price: ${predicted_price:.2f}")
        print(f"✅ Predicted Change: {predicted_change_pct:.2f}%")
        print(f"✅ Reasoning: {reasoning[:100]}...")
        
        # Generate reflection
        print("Generating reflection...")
        reflection = generate_reflection(
            row['target_date'],
            row['target_price'],
            predicted_price,
            row['actual_price'],
            row['summarized_context'],
            reasoning
        )
        
        if reflection is None:
            print(f"❌ Failed to generate reflection for week {i+1}")
            continue
        
        print(f"✅ Reflection: {reflection[:100]}...")
        
        # Update the dataframe with proper dtype handling
        df.loc[i, 'predicted_price'] = predicted_price
        df.loc[i, 'predicted_change_pct'] = predicted_change_pct
        df.loc[i, 'llm_reasoning'] = str(reasoning)
        df.loc[i, 'llm_reflection'] = str(reflection)
        
        # Save after each week to avoid losing progress
        df.to_csv('training_set.csv', index=False)
        
        # Summary for this week
        direction_correct = ('YES' if (predicted_change_pct > 0 and row['actual_change_pct'] > 0) or (predicted_change_pct < 0 and row['actual_change_pct'] < 0) else 'NO')
        print(f"📊 WEEK SUMMARY:")
        print(f"   Predicted: ${predicted_price:.2f} ({predicted_change_pct:.2f}%)")
        print(f"   Actual: ${row['actual_price']:.2f} ({row['actual_change_pct']:.2f}%)")
        print(f"   Direction Correct: {direction_correct}")
        
        processed_count += 1
        success_count += 1
        
        print(f"✅ Week {i+1} completed successfully!")
        print("=" * 60)
        
        # Add a small delay to avoid rate limiting
        time.sleep(2)
    
    print(f"\n🎉 FINAL SUMMARY:")
    print(f"   Total weeks: {total_weeks}")
    print(f"   Processed: {processed_count}")
    print(f"   Successful: {success_count}")
    print(f"   Success rate: {(success_count/processed_count)*100:.1f}%" if processed_count > 0 else "   Success rate: 0%")
    
    return success_count > 0

if __name__ == "__main__":
    print("Starting Phase 3: LLM Prediction Generation")
    print("Processing all weeks...")
    print("\n")
    
    if process_all_weeks():
        print("\n🎉 All weeks processing completed successfully!")
    else:
        print("\n❌ Processing failed or no weeks were processed.") 