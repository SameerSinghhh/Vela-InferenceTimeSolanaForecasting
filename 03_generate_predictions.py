import os
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system
except ImportError:
    print("Error: xai_sdk not installed. Please install with: pip install xai-sdk")
    exit(1)

# Initialize Grok client
client = Client(api_key=os.getenv('GROK_API_KEY'))

def generate_prediction(target_date, current_sol_price, prediction_date, tweets, market_data):
    """
    Generate SOL price prediction using Grok based on current price, tweets, and previous week's market data.
    NO LOOKAHEAD BIAS - only uses data available on or before target_date.
    """
    
    # Convert dates to readable format
    target_date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%B %d, %Y')
    prediction_date_str = datetime.strptime(prediction_date, '%Y-%m-%d').strftime('%B %d, %Y')
    
    prediction_prompt = f"""You are a professional cryptocurrency analyst making a Solana (SOL) price prediction.

STRICT NO LOOKAHEAD RULE: You can ONLY use information available on or before {target_date_str}. 

CURRENT MARKET DATA (as of {target_date_str}):
- Current SOL Price: ${current_sol_price:.2f}
- Prediction Target: {prediction_date_str} (1 day ahead)

MARKET CONTEXT (as of {target_date_str}):
- BTC 30-day change: {market_data['btc_30d_change']:.2f}%
- ETH 30-day change: {market_data['eth_30d_change']:.2f}%  
- SOL 30-day change: {market_data['sol_30d_change']:.2f}%
- S&P 500 30-day change: {market_data['sp_30d_change']:.2f}%

RECENT PRICE TRENDS (Context Period):
- BTC Prices: {market_data['btc_prices']}
- ETH Prices: {market_data['eth_prices']}
- SOL Prices: {market_data['sol_prices']}
- S&P 500 Prices: {market_data['sp_prices']}

MARKET SENTIMENT & NEWS (from social media analysis):
{tweets}

ANALYSIS INSTRUCTIONS:
1. Analyze the recent crypto market trends from the context period
2. Consider correlation with traditional markets (S&P 500)
3. Evaluate sentiment from the social media analysis
4. Factor in SOL's 30-day performance vs other assets
5. Make a realistic 1-day price prediction

RESPONSE FORMAT:
PREDICTED_PRICE: [dollar amount, e.g. 185.50]
REASONING: [3-4 sentences explaining your analysis and prediction rationale]

Provide a specific SOL price prediction for {prediction_date_str}."""

    try:
        # Create chat with Grok
        chat = client.chat.create(
            model="grok-4-0709",
            temperature=0.2
        )
        
        # Add system message
        chat.append(system("You are a professional cryptocurrency analyst specializing in Solana (SOL) price prediction. You provide data-driven analysis with clear reasoning."))
        
        # Add user prompt
        chat.append(user(prediction_prompt))
        
        # Get response
        response = chat.sample()
        content = response.content.strip()
        
        # Parse the response
        import re
        predicted_price = None
        reasoning = ""
        
        # Extract predicted price
        price_patterns = [
            r'PREDICTED_PRICE:\s*[\$]?(\d+\.?\d*)',
            r'prediction.*?[\$]?(\d+\.?\d*)',
            r'target.*?[\$]?(\d+\.?\d*)',
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
            r'REASONING:\s*(.+?)(?:\n\n|\n[A-Z]|\Z)',
            r'reasoning.*?:\s*(.+?)(?:\n\n|\n[A-Z]|\Z)',
            r'analysis.*?:\s*(.+?)(?:\n\n|\n[A-Z]|\Z)'
        ]
        
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                reasoning = matches[0].strip()
                break
        
        # Fallback: use whole response if no specific reasoning found
        if not reasoning:
            reasoning = content.strip()
        
        # Validate price is reasonable
        if predicted_price is None:
            # Try to extract any reasonable SOL price from response
            all_numbers = re.findall(r'\d+\.?\d*', content)
            if all_numbers:
                numbers = [float(n) for n in all_numbers if 50 < float(n) < 1000]
                if numbers:
                    predicted_price = numbers[0]
        
        if predicted_price is None:
            raise ValueError(f"Could not parse predicted price from: {content}")
        
        return predicted_price, reasoning
        
    except Exception as e:
        print(f"Error generating prediction: {e}")
        return None, None

def generate_reflection(target_date, prediction_date, current_price, predicted_price, actual_price, reasoning):
    """
    Generate a reflection on why the prediction was right or wrong.
    CRITICAL: NO LOOKAHEAD BIAS - analyze only with information available at prediction time.
    """
    
    predicted_change = ((predicted_price - current_price) / current_price) * 100
    actual_change = ((actual_price - current_price) / current_price) * 100
    
    target_date_str = datetime.strptime(target_date, '%Y-%m-%d').strftime('%B %d, %Y')
    prediction_date_str = datetime.strptime(prediction_date, '%Y-%m-%d').strftime('%B %d, %Y')
    
    direction_correct = (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0)
    
    reflection_prompt = f"""CRITICAL: Analyze this SOL prediction using ONLY information that was available on {target_date_str}. NO HINDSIGHT BIAS.

PREDICTION ANALYSIS:
- Analysis Date: {target_date_str}
- Target Date: {prediction_date_str}
- Starting Price: ${current_price:.2f}
- Predicted Price: ${predicted_price:.2f} ({predicted_change:.1f}%)
- Actual Price: ${actual_price:.2f} ({actual_change:.1f}%)
- Direction Accuracy: {"CORRECT" if direction_correct else "INCORRECT"}
- Price Accuracy: {abs(predicted_price - actual_price):.2f} difference

ORIGINAL REASONING:
{reasoning}

REFLECTION TASK:
Analyze why this prediction succeeded or failed. Consider:
1. Quality of the available market data and analysis
2. Effectiveness of the reasoning methodology  
3. Impact of crypto market volatility/unpredictability
4. Specific factors that may have been overlooked or overweighted

IMPORTANT: Do NOT use hindsight knowledge. Only reference factors that were knowable on {target_date_str}.

Provide a concrete, specific 2-3 sentence analysis of what went right or wrong with this prediction approach."""

    try:
        # Create chat with Grok
        chat = client.chat.create(
            model="grok-4-0709",
            temperature=0.3
        )
        
        # Add system message
        chat.append(system("You are an objective analyst providing reflections on cryptocurrency predictions. You focus on methodology and avoid hindsight bias."))
        
        # Add user prompt
        chat.append(user(reflection_prompt))
        
        # Get response
        response = chat.sample()
        return response.content.strip()
        
    except Exception as e:
        print(f"Error generating reflection: {e}")
        return None

def needs_prediction(row):
    """Check if a week needs prediction (any required column is empty)"""
    required_cols = ['predicted_price', 'predicted_change_pct', 'llm_reasoning', 'llm_reflection']
    return any(pd.isna(row[col]) or str(row[col]).strip() == '' for col in required_cols)

def get_current_week_market_data(df, current_week_idx):
    """Get current week's market data available at prediction time (NO LOOKAHEAD BIAS)"""
    current_week = df.iloc[current_week_idx]
    return {
        'btc_30d_change': current_week.get('btc_30d_change_pct', 0.0),
        'eth_30d_change': current_week.get('eth_30d_change_pct', 0.0),
        'sol_30d_change': current_week.get('sol_30d_change_pct', 0.0),
        'sp_30d_change': current_week.get('sp500_30d_change_pct', 0.0),
        'btc_prices': str(current_week.get('btc_price_array', '')),
        'eth_prices': str(current_week.get('eth_price_array', '')),
        'sol_prices': str(current_week.get('sol_price_array', '')),
        'sp_prices': str(current_week.get('sp_price_array', ''))
    }

def process_training_weeks():
    """
    Process ALL weeks to generate predictions with NO MEMORY.
    Only completes weeks that haven't been done yet.
    """
    
    # Load the final training data
    df = pd.read_csv('final_training_set.csv')
    
    # Process ALL weeks in the dataset
    weeks_to_process = len(df)
    
    print(f"🚀 GENERATING PREDICTIONS (NO MEMORY)")
    print(f"📊 Processing ALL {weeks_to_process} weeks in final_training_set.csv")
    print(f"🎯 Training weeks 1-40: (no STM/LTM memory)")
    print(f"🎯 Test weeks 41-{weeks_to_process}: (no STM/LTM memory)")
    print(f"⚠️  Only completing weeks that haven't been processed yet")
    print("=" * 60)
    
    processed_count = 0
    success_count = 0
    
    for i in range(weeks_to_process):
        row = df.iloc[i]
        week_num = i + 1
        
        # Skip if already has predictions
        if not needs_prediction(row):
            print(f"Week {week_num}: Already has predictions - SKIPPING")
            continue
        
        print(f"\n🧠 PROCESSING WEEK {week_num} (Training - No Memory)")
        print(f"📅 Target Date: {row['target_date']}")
        print(f"📅 Prediction Date: {row['prediction_date']}")
        print(f"💰 Current SOL Price: ${row['target_date_sol_price']:.2f}")
        print(f"💰 Actual SOL Price: ${row['prediction_date_sol_price']:.2f}")
        print(f"📊 Actual Change: {row['actual_change_pct']:.2f}%")
        
        # Get current week's market data (available at target_date - no lookahead bias)
        market_data = get_current_week_market_data(df, i)
        
        # Get current week's tweets
        tweets = row.get('tweets', 'No tweet data available')
        
        print(f"📱 Tweet data length: {len(str(tweets))} chars")
        print(f"📈 Market context (as of target date): BTC {market_data['btc_30d_change']:.1f}%, ETH {market_data['eth_30d_change']:.1f}%, SOL {market_data['sol_30d_change']:.1f}%")
        print("\n" + "-"*50)
        
        # Generate prediction
        print("🤖 Generating prediction with Grok...")
        predicted_price, reasoning = generate_prediction(
            row['target_date'],
            row['target_date_sol_price'],
            row['prediction_date'],
            tweets,
            market_data
        )
        
        if predicted_price is None:
            print(f"❌ Failed to generate prediction for week {week_num}")
            continue
        
        # Calculate predicted change percentage
        predicted_change_pct = ((predicted_price - row['target_date_sol_price']) / row['target_date_sol_price']) * 100
        
        print(f"✅ Predicted Price: ${predicted_price:.2f}")
        print(f"✅ Predicted Change: {predicted_change_pct:.2f}%")
        print(f"✅ Reasoning: {reasoning[:120]}...")
        
        # Generate reflection
        print("🔍 Generating reflection...")
        reflection = generate_reflection(
            row['target_date'],
            row['prediction_date'],
            row['target_date_sol_price'],
            predicted_price,
            row['prediction_date_sol_price'],
            reasoning
        )
        
        if reflection is None:
            print(f"❌ Failed to generate reflection for week {week_num}")
            continue
        
        print(f"✅ Reflection: {reflection[:120]}...")
        
        # Update the dataframe (SAFE - only update empty columns)
        df.at[i, 'predicted_price'] = predicted_price
        df.at[i, 'predicted_change_pct'] = predicted_change_pct
        df.at[i, 'llm_reasoning'] = str(reasoning)
        df.at[i, 'llm_reflection'] = str(reflection)
        
        # Save immediately to avoid losing progress
        df.to_csv('final_training_set.csv', index=False)
        
        # Week summary
        direction_correct = (predicted_change_pct > 0 and row['actual_change_pct'] > 0) or (predicted_change_pct < 0 and row['actual_change_pct'] < 0)
        print(f"\n📊 WEEK {week_num} SUMMARY:")
        print(f"   🎯 Predicted: ${predicted_price:.2f} ({predicted_change_pct:.2f}%)")
        print(f"   🎯 Actual: ${row['prediction_date_sol_price']:.2f} ({row['actual_change_pct']:.2f}%)")
        print(f"   🎯 Direction: {'✅ CORRECT' if direction_correct else '❌ INCORRECT'}")
        print(f"   🎯 Price Error: ${abs(predicted_price - row['prediction_date_sol_price']):.2f}")
        
        processed_count += 1
        success_count += 1
        
        print(f"✅ Week {week_num} completed successfully!")
        print("=" * 60)
        
        # Delay to avoid rate limiting
        time.sleep(3)
    
    print(f"\n🎉 PREDICTION GENERATION SUMMARY:")
    print(f"   📊 Weeks processed: {processed_count}")
    print(f"   ✅ Successful: {success_count}")
    print(f"   📈 Success rate: {(success_count/processed_count)*100:.1f}%" if processed_count > 0 else "   📈 Success rate: 0%")
    
    return success_count > 0

if __name__ == "__main__":
    print("🚀 PHASE 3: PREDICTION GENERATION (NO MEMORY)")
    print("🎯 Generating predictions for ALL weeks (NO MEMORY)")
    print("📊 Processing all weeks in final_training_set.csv...")
    print("⚠️  Only completing missing predictions, preserving existing data")
    print("\n")
    
    if process_training_weeks():
        print(f"\n🎉 All weeks processed successfully!")
        print("📝 All predictions in final_training_set.csv are now complete")
    else:
        print("\n❌ Processing failed or no weeks were processed.") 