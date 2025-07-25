#!/usr/bin/env python3
"""
Process raw tweets using Grok LLM to select most impactful tweets.
Updates final_training_set.csv tweets column with formatted results.
"""

import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional

try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system
except ImportError:
    print("Error: xai_sdk not installed. Please install with: pip install xai-sdk")
    exit(1)

class TweetProcessor:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize Grok client
        api_key = self.get_api_key()
        self.client = Client(api_key=api_key)
        
        self.assets = {
            'BTC': 'BITCOIN',
            'ETH': 'ETHEREUM', 
            'SOL': 'SOLANA',
            'SP500': 'S&P 500'
        }
        
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load both CSV files safely"""
        try:
            training_df = pd.read_csv('final_training_set.csv')
            raw_tweets_df = pd.read_csv('raw_tweets_data.csv')
            print(f"✅ Loaded final_training_set.csv: {len(training_df)} weeks")
            print(f"✅ Loaded raw_tweets_data.csv: {len(raw_tweets_df)} weeks")
            return training_df, raw_tweets_df
        except Exception as e:
            print(f"❌ Error loading CSV files: {e}")
            return None, None
    
    def needs_tweets(self, training_df: pd.DataFrame, week_idx: int) -> bool:
        """Check if a week needs tweets (tweets column is empty)"""
        tweets_value = training_df.iloc[week_idx]['tweets']
        return pd.isna(tweets_value) or str(tweets_value).strip() == ''
    
    def get_raw_tweets_for_week(self, raw_tweets_df: pd.DataFrame, week_idx: int) -> Dict[str, str]:
        """Get raw tweet data for all assets for a specific week"""
        week_tweets = {}
        for asset in self.assets.keys():
            col = f'{asset}_tweets'
            raw_data = raw_tweets_df.iloc[week_idx][col]
            
            if pd.notna(raw_data) and str(raw_data).strip():
                week_tweets[asset] = str(raw_data)
            else:
                week_tweets[asset] = None
                
        return week_tweets
    
    def call_grok_api(self, prompt: str) -> Optional[str]:
        """Call Grok API using xai_sdk"""
        try:
            # Create chat with search mode OFF
            chat = self.client.chat.create(
                model="grok-4-0709",
                temperature=0.1
            )
            
            # Add system message for financial analysis expertise
            chat.append(system("You are an expert financial analyst who identifies the most impactful cryptocurrency and market tweets that could influence price movements."))
            
            # Add user prompt
            chat.append(user(prompt))
            
            # Get response
            response = chat.sample()
            return response.content
            
        except Exception as e:
            print(f"❌ Error calling Grok API: {e}")
            return None
    
    def get_api_key(self) -> str:
        """Get API key from .env file"""
        api_key = os.getenv('GROK_API_KEY')
        if not api_key:
            raise ValueError("GROK_API_KEY not found in .env file")
        return api_key
    
    def create_grok_prompt(self, week_tweets: Dict[str, str], start_date: str, end_date: str) -> str:
        """Create the prompt for Grok to analyze tweets"""
        prompt = f"""
TASK: Analyze raw tweet data and select the 2 MOST IMPACTFUL tweets per asset that could influence price movements.

DATE RANGE: {start_date} to {end_date}

CRITERIA FOR SELECTION:
- Tweets that contain breaking news, major announcements, or significant market developments
- High engagement (likes, retweets, views) indicating market attention
- Content that could directly impact asset prices (adoption news, regulatory updates, technical developments, institutional moves)
- Tweets from influential accounts or verified sources
- Avoid duplicate information - select diverse, complementary insights

RAW TWEET DATA:
"""
        
        for asset, raw_data in week_tweets.items():
            if raw_data:
                prompt += f"\n=== {self.assets[asset]} ({asset}) RAW DATA ===\n{raw_data}\n"
            else:
                prompt += f"\n=== {self.assets[asset]} ({asset}) RAW DATA ===\nNo data available\n"
        
        prompt += f"""

OUTPUT FORMAT:
For each asset, provide exactly 2 tweets in this format:

=== BITCOIN (BTC) ===
@username | Tweet text | YYYY-MM-DD | Faves: X | RTs: Y
@username | Tweet text | YYYY-MM-DD | Faves: X | RTs: Y

=== ETHEREUM (ETH) ===  
@username | Tweet text | YYYY-MM-DD | Faves: X | RTs: Y
@username | Tweet text | YYYY-MM-DD | Faves: X | RTs: Y

=== SOLANA (SOL) ===
@username | Tweet text | YYYY-MM-DD | Faves: X | RTs: Y  
@username | Tweet text | YYYY-MM-DD | Faves: X | RTs: Y

=== S&P 500 (SP500) ===
@username | Tweet text | YYYY-MM-DD | Faves: X | RTs: Y
@username | Tweet text | YYYY-MM-DD | Faves: X | RTs: Y

IMPORTANT:
- Extract actual usernames, tweet text, dates, likes (faves), and retweets from the raw data
- If an asset has no data, still include the header but put "No impactful tweets found" 
- Focus on tweets most likely to move markets or indicate significant developments
- Prioritize breaking news, institutional adoption, regulatory updates, major partnerships
"""
        return prompt
    
    def process_week(self, training_df: pd.DataFrame, raw_tweets_df: pd.DataFrame, week_idx: int) -> bool:
        """Process a single week's tweets"""
        week_num = week_idx + 1
        start_date = training_df.iloc[week_idx]['context_start_date']
        end_date = training_df.iloc[week_idx]['prediction_date']
        
        print(f"\n" + "="*60)
        print(f"🧠 PROCESSING WEEK {week_num}: {start_date} to {end_date}")
        
        # Get raw tweets for this week
        week_tweets = self.get_raw_tweets_for_week(raw_tweets_df, week_idx)
        
        # Check if we have any data
        has_data = any(tweets is not None for tweets in week_tweets.values())
        if not has_data:
            print(f"⚠️  No raw tweet data available for Week {week_num}")
            return False
        
        # Create Grok prompt
        prompt = self.create_grok_prompt(week_tweets, start_date, end_date)
        
        print(f"🤖 Calling Grok API to analyze tweets...")
        
        # Call Grok API
        grok_response = self.call_grok_api(prompt)
        
        if grok_response:
            # Update the training DataFrame
            training_df.at[week_idx, 'tweets'] = grok_response
            print(f"✅ Grok analysis complete for Week {week_num}")
            
            # Save the file immediately
            training_df.to_csv('final_training_set.csv', index=False)
            print(f"💾 Updated final_training_set.csv")
            
            return True
        else:
            print(f"❌ Failed to get Grok response for Week {week_num}")
            return False
    
    def process_all_weeks(self):
        """Process all weeks that need tweets"""
        # Load data
        training_df, raw_tweets_df = self.load_data()
        if training_df is None or raw_tweets_df is None:
            return
        
        # Find weeks that need processing
        weeks_to_process = []
        for week_idx in range(len(training_df)):
            if self.needs_tweets(training_df, week_idx):
                weeks_to_process.append(week_idx)
        
        total_weeks = len(weeks_to_process)
        if total_weeks == 0:
            print("🎉 All weeks already have tweets! Nothing to process.")
            return
        
        print(f"\n🐦 STARTING TWEET PROCESSING WITH GROK")
        print(f"📊 Found {total_weeks} weeks that need tweets")
        print(f"⏰ Estimated time: {total_weeks * 30} seconds (30s per week)")
        
        processed = 0
        failed = 0
        
        # Process each week
        for i, week_idx in enumerate(weeks_to_process):
            try:
                success = self.process_week(training_df, raw_tweets_df, week_idx)
                if success:
                    processed += 1
                else:
                    failed += 1
                    
                # Small delay between API calls
                if i < len(weeks_to_process) - 1:  # Don't delay after last call
                    print(f"⏳ Waiting 5 seconds before next week...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"❌ Error processing week {week_idx + 1}: {e}")
                failed += 1
        
        print(f"\n" + "="*60)
        print(f"📋 PROCESSING COMPLETE!")
        print(f"   ✅ Processed: {processed} weeks")
        print(f"   ❌ Failed: {failed} weeks")
        print(f"   📈 Success rate: {processed/(processed+failed)*100:.1f}%")

def main():
    print("🚀 STARTING TWEET PROCESSING WITH GROK LLM")
    print("=" * 60)
    
    try:
        processor = TweetProcessor()
        print(f"✅ Grok client initialized successfully")
        
        processor.process_all_weeks()
        
        print(f"\n🎉 SUCCESS! Tweets processed and added to final_training_set.csv")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("❌ Please ensure your .env file contains GROK_API_KEY")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return

if __name__ == "__main__":
    main() 