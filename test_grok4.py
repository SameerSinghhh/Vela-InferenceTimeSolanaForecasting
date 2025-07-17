#!/usr/bin/env python3
"""
Generate tweets for all weeks in the training dataset using Grok-4
"""

import sys
import pandas as pd
from datetime import datetime
import time
import os

try:
    from xai_sdk import Client
    from xai_sdk.chat import user
    from xai_sdk.search import SearchParameters, x_source
except ImportError:
    print("Error: xai_sdk not installed. Please install with: pip install xai-sdk")
    sys.exit(1)

# API key from environment variable
API_KEY = os.getenv('GROK_API_KEY')
if not API_KEY:
    print("Error: GROK_API_KEY environment variable not set.")
    print("Please run: export GROK_API_KEY='your-api-key-here'")
    sys.exit(1)

def get_tweets_for_period(client, context_start_date, target_date):
    """Get tweets for a specific date period"""
    
    # Convert string dates to datetime objects
    start_date = pd.to_datetime(context_start_date)
    end_date = pd.to_datetime(target_date)
    
    # Create chat with search parameters for this specific period
    chat = client.chat.create(
        model="grok-4-0709",
        search_parameters=SearchParameters(
            mode="on",  # Force live search
            sources=[x_source(post_view_count=5000)],  # Lower threshold for better historical coverage
            return_citations=False,  # No citations needed
            from_date=start_date,
            to_date=end_date,
            max_search_results=25  # Under 30 limit
        )
    )
    
    # Dynamic prompt with the specific date range
    prompt = f"""
Find high-quality financial news tweets STRICTLY from {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')} about:
- Solana (SOL)
- Ethereum (ETH)
- Bitcoin (BTC)
- S&P 500

🚨 CRITICAL DATE VERIFICATION REQUIREMENTS:
- ONLY include tweets that are DEFINITIVELY from the exact date range {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}
- If you cannot find authentic tweets from this specific time period for any asset, respond with "INSUFFICIENT DATA FOR [ASSET]" instead
- DO NOT use tweets from any other dates, even if they are highly relevant
- WE PREFER FEWER CORRECT TWEETS OVER ANY TWEETS FROM WRONG DATES
- Aim for 3 tweets per asset (12 total) but accept fewer if that's all that exists in the correct date range

Content Requirements:
- Must be HIGH-IMPACT tweets that could influence asset prices or market sentiment
- Each tweet must be UNIQUE - no duplicates across any categories
- Each tweet must be in English
- INCLUDE news regardless of whether it is POSITIVE, NEGATIVE, or NEUTRAL — if it could have a big impact, include it
- NO giveaways, spam, or purely speculative non-informative content
- Focus on news, analysis, price movements, institutional adoption, regulatory updates, market analysis, or significant market events

Format exactly like this, segmented by asset:

=== SOLANA (SOL) ===
@username | Tweet text | Time | Views: X

=== ETHEREUM (ETH) ===
@username | Tweet text | Time | Views: X

=== BITCOIN (BTC) ===
@username | Tweet text | Time | Views: X

=== S&P 500 ===
@username | Tweet text | Time | Views: X

REMEMBER: Better to return "INSUFFICIENT DATA" than incorrect timestamps. Only return tweets you are 100% certain are from {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}.
"""
    
    chat.append(user(prompt))
    response = chat.sample()
    return response.content

def process_training_dataset():
    """Process all weeks in the training dataset and populate tweets"""
    
    # Read the training dataset
    print("Loading training dataset...")
    df = pd.read_csv('training_set2_with_btc_eth_sol_sp.csv')
    
    # Initialize Grok-4 client
    client = Client(api_key=API_KEY)
    
    print(f"Found {len(df)} periods to process")
    
    for index, row in df.iterrows():
        period_num = index + 1
        context_start = row['context_start_date']
        target_date = row['target_date']
        current_tweets = row['tweets']
        
        # Skip if tweets already filled in
        if pd.notna(current_tweets) and str(current_tweets).strip():
            print(f"Period {period_num}: {context_start} to {target_date} - SKIPPING (tweets already exist)")
            continue
        
        print(f"\nPeriod {period_num}: {context_start} to {target_date} - PROCESSING...")
        
        # Retry logic for failed requests
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Get tweets for this period
                tweets_response = get_tweets_for_period(client, context_start, target_date)
                
                # Update the dataframe with the tweets
                df.at[index, 'tweets'] = tweets_response
                
                # Save the CSV file after each completion
                df.to_csv('training_set2_with_btc_eth_sol_sp.csv', index=False)
                
                print(f"✅ Period {period_num} completed and saved")
                print(f"Preview: {tweets_response[:100]}...")
                success = True
                
                # Brief pause to avoid rate limits
                time.sleep(2)
                
            except Exception as e:
                retry_count += 1
                print(f"❌ Period {period_num} failed (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"❌ Period {period_num} failed after {max_retries} attempts. Stopping.")
                    return
    
    print(f"\n🎉 All periods processed successfully!")

if __name__ == "__main__":
    process_training_dataset() 