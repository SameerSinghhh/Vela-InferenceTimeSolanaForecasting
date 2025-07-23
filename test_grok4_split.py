#!/usr/bin/env python3
"""
Generate tweets for all weeks in the training dataset using Grok-4
Split into 4 separate calls per period - one for each asset
"""

import sys
import pandas as pd
from datetime import datetime
import time
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from xai_sdk import Client
    from xai_sdk.chat import user
    from xai_sdk.search import SearchParameters, x_source
except ImportError:
    print("Error: xai_sdk not installed. Please install with: pip install xai-sdk")
    sys.exit(1)

# API key from .env file
API_KEY = os.getenv('GROK_API_KEY')
if not API_KEY:
    print("Error: GROK_API_KEY not found in .env file.")
    print("Please add GROK_API_KEY=your-api-key to your .env file")
    sys.exit(1)

def validate_tweet_dates(tweets_text, start_date, end_date):
    """Ultra-strict validation that all tweets are within the correct date range (inclusive of target_date)"""
    import re
    from datetime import datetime
    
    lines = tweets_text.split('\n')
    valid_tweets = []
    invalid_tweets = []
    
    print(f"      🔍 VALIDATING DATE RANGE: {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')} (INCLUSIVE)")
    print(f"      📅 Valid dates: {start_date.strftime('%b %d')}, {(start_date + pd.Timedelta(days=1)).strftime('%b %d') if (end_date - start_date).days > 0 else ''}, {end_date.strftime('%b %d')}")
    
    for line in lines:
        if '@' in line and '|' in line:  # This looks like a tweet line
            # Extract date from various formats in the line
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',  # 2024-07-30T14:21:20
                r'(\d{4}-\d{2}-\d{2})',                      # 2024-07-30
                r'(\w{3} \d{1,2}, \d{4})',                   # Jul 30, 2024
                r'(\d{1,2}/\d{1,2}/\d{4})',                  # 7/30/2024
            ]
            
            tweet_date = None
            date_str_found = None
            for pattern in date_patterns:
                match = re.search(pattern, line)
                if match:
                    date_str_found = match.group(1)
                    try:
                        if 'T' in date_str_found:
                            tweet_date = datetime.fromisoformat(date_str_found.replace('Z', '+00:00')).replace(tzinfo=None)
                        elif '-' in date_str_found and len(date_str_found) == 10:
                            tweet_date = datetime.strptime(date_str_found, '%Y-%m-%d')
                        elif '/' in date_str_found:
                            tweet_date = datetime.strptime(date_str_found, '%m/%d/%Y')
                        elif ',' in date_str_found:
                            tweet_date = datetime.strptime(date_str_found, '%b %d, %Y')
                        break
                    except Exception as e:
                        print(f"      ⚠️ Date parsing error for '{date_str_found}': {e}")
                        continue
            
            if tweet_date:
                # Check if tweet date falls within our range (INCLUSIVE of both start and end dates)
                if start_date.date() <= tweet_date.date() <= end_date.date():
                    valid_tweets.append(line)
                    print(f"      ✅ VALID: {tweet_date.strftime('%b %d, %Y')} (within range) - @{line.split('|')[0].strip().replace('@', '')}")
                else:
                    invalid_tweets.append(line)
                    if tweet_date.date() < start_date.date():
                        days_off = (start_date.date() - tweet_date.date()).days
                        print(f"      ❌ TOO EARLY: {tweet_date.strftime('%b %d, %Y')} ({days_off} days before start) - REJECTED")
                    else:
                        days_off = (tweet_date.date() - end_date.date()).days
                        print(f"      ❌ TOO LATE: {tweet_date.strftime('%b %d, %Y')} ({days_off} days after target) - REJECTED")
            else:
                # If we can't parse the date, reject it entirely  
                print(f"      ⚠️ NO DATE FOUND - REJECTED: {line[:80]}...")
        else:
            # Keep headers and formatting lines
            valid_tweets.append(line)
    
    valid_count = len([line for line in valid_tweets if '@' in line])
    invalid_count = len(invalid_tweets)
    
    print(f"      📊 FINAL RESULTS: {valid_count} valid tweets, {invalid_count} rejected tweets")
    
    return '\n'.join(valid_tweets), valid_count

def get_tweets_for_asset(client, asset_name, asset_symbol, context_start_date, target_date):
    """Get tweets using simple X query + strict validation loop"""
    
    # Convert string dates to datetime objects for X search format
    start_date = pd.to_datetime(context_start_date)
    end_date = pd.to_datetime(target_date)
    
    # Create asset-specific search terms
    search_terms = {
        "Solana": "(solana OR $SOL OR #Solana)",
        "Ethereum": "(ethereum OR $ETH OR #Ethereum)", 
        "Bitcoin": "(bitcoin OR $BTC OR #Bitcoin)",
        "S&P 500": "(SPX OR \"S&P 500\" OR SP500)"
    }
    
    # Get search terms for this asset
    asset_terms = search_terms.get(asset_name, f"({asset_name} OR ${asset_symbol})")
    
    # Create exact X search query
    x_search_query = f"{asset_terms} since:{start_date.strftime('%Y-%m-%d')} until:{end_date.strftime('%Y-%m-%d')} (min_faves:0 OR min_retweets:0)"
    
    print(f"    🔍 X Search Query: {x_search_query}")
    
    valid_tweets = []
    attempts = 0
    max_attempts = 10  # Keep trying until we get enough valid tweets
    
    while len(valid_tweets) < 2 and attempts < max_attempts:
        attempts += 1
        print(f"    Attempt {attempts}: Simple X query approach")
        
        # Remove all date parameters - let prompt handle everything like web Grok
        chat = client.chat.create(
            model="grok-4-0709",
            search_parameters=SearchParameters(
                mode="on",  # Force live search
                sources=[x_source()],  # X source
                return_citations=False
            )
        )
        
        # Cache-busting search with very specific date targeting
        specific_dates = []
        current_date = start_date
        while current_date <= end_date:
            specific_dates.append(current_date.strftime('%B %d, %Y'))
            current_date += pd.Timedelta(days=1)
        
        date_list = " and ".join(specific_dates)
        
        prompt = f"""
FRESH SEARCH REQUEST #{attempts} for {asset_name} ({asset_symbol}):

I need tweets posted SPECIFICALLY on these exact dates: {date_list}

Search X for tweets about {asset_name} from these precise days only:
{chr(10).join([f"- {date}" for date in specific_dates])}

Find 20 different tweets from this specific time period. Do NOT reuse previous results.

Format:
=== {asset_name.upper()} ({asset_symbol}) ===
@username | Tweet text | YYYY-MM-DD | Faves: X | RTs: Y

This is search attempt #{attempts}. Return NEW tweets from the specified dates only.
"""
        
        chat.append(user(prompt))
        response = chat.sample()
        response_text = response.content.strip()
        
        if "@" in response_text:
            # Validate dates and extract only valid tweets
            validated_text, valid_count = validate_tweet_dates(response_text, start_date, end_date)
            
            if valid_count > 0:
                # Extract individual valid tweets to add to our collection
                lines = validated_text.split('\n')
                for line in lines:
                    if '@' in line and '|' in line and line not in valid_tweets:
                        valid_tweets.append(line)
                        if len(valid_tweets) >= 2:
                            break
                
                print(f"    ✅ Found {valid_count} valid tweets this attempt (total: {len(valid_tweets)}) out of ~20 requested")
            else:
                print(f"    ❌ No valid tweets found this attempt out of ~20 requested, trying again...")
        else:
            print(f"    ❌ No tweets returned, trying again...")
        
        time.sleep(1)
    
    if len(valid_tweets) >= 2:
        # Format the final result
        result = f"=== {asset_name.upper()} ({asset_symbol}) ===\n"
        result += '\n'.join(valid_tweets[:3])  # Take top 3 valid tweets
        print(f"    🎉 SUCCESS! Collected {len(valid_tweets[:3])} valid tweets for {asset_name} (from {attempts} attempts of 20 tweets each)")
        return result
    else:
        print(f"    ⚠️ Could not find enough valid tweets for {asset_name} after {attempts} attempts of 20 tweets each")
        return f"=== {asset_name.upper()} ({asset_symbol}) ===\nInsufficient valid tweets found for date range"

def get_tweets_for_period(client, context_start_date, target_date):
    """Get tweets for all 4 assets in a specific date period"""
    
    assets = [
        ("Solana", "SOL"),
        ("Ethereum", "ETH"), 
        ("Bitcoin", "BTC"),
        ("S&P 500", "SPX")
    ]
    
    combined_response = ""
    
    for asset_name, asset_symbol in assets:
        print(f"  Getting {asset_name} ({asset_symbol}) tweets...")
        
        try:
            asset_response = get_tweets_for_asset(client, asset_name, asset_symbol, context_start_date, target_date)
            combined_response += asset_response + "\n\n"
            print(f"  ✅ {asset_name} processing completed")
            
        except Exception as e:
            print(f"  ❌ {asset_name} failed with error: {e}")
            # Add error message but continue with other assets
            combined_response += f"=== {asset_name.upper()} ({asset_symbol}) ===\nERROR PROCESSING {asset_name.upper()} ({asset_symbol}): {str(e)}\n\n"
    
    return combined_response.strip()

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
        
        try:
            # Get tweets for all 4 assets in this period
            tweets_response = get_tweets_for_period(client, context_start, target_date)
            
            # Update the dataframe with the tweets
            df.at[index, 'tweets'] = tweets_response
            
            # Save the CSV file after each completion
            df.to_csv('training_set2_with_btc_eth_sol_sp.csv', index=False)
            
            print(f"✅ Period {period_num} completed and saved")
            print(f"Preview: {tweets_response[:150]}...")
            
            # Brief pause between periods
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Period {period_num} failed: {e}")
            print("Stopping processing.")
            return
    
    print(f"\n🎉 All periods processed successfully!")

if __name__ == "__main__":
    process_training_dataset() 