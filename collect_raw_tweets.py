#!/usr/bin/env python3
"""
Collect raw tweets and store them in a separate CSV file.
Creates raw_tweets_data.csv with all 51 weeks, populates first week only.
"""

import pandas as pd
import requests
import time
import random
import json
import urllib.parse
from typing import List, Dict, Optional

class RawTweetCollector:
    def __init__(self):
        self.api_url = "https://x.vela.partners/search"
        self.assets = ['BTC', 'ETH', 'SOL', 'SP500']
        self.asset_queries = {
            'BTC': '(bitcoin OR $BTC OR #Bitcoin)',
            'ETH': '(ethereum OR $ETH OR #Ethereum)', 
            'SOL': '(solana OR $SOL OR #Solana)',
            'SP500': '(S&P500 OR #SP500 OR "S&P 500")'
        }
        
    def load_existing_csv(self) -> pd.DataFrame:
        """Load existing CSV file or create if doesn't exist"""
        try:
            df = pd.read_csv('raw_tweets_data.csv')
            print(f"📊 Loaded existing CSV with {len(df)} weeks")
            return df
        except FileNotFoundError:
            print("📊 CSV not found, creating new structure...")
            # Load training data to get all date ranges
            df_training = pd.read_csv('final_training_set.csv')
            
            # Create new dataframe structure
            columns = ['start_date', 'end_date'] + [f'{asset}_tweets' for asset in self.assets]
            data = []
            
            for idx, row in df_training.iterrows():
                week_data = {
                    'start_date': row['context_start_date'],
                    'end_date': row['prediction_date'],
                    'BTC_tweets': '',
                    'ETH_tweets': '',
                    'SOL_tweets': '',
                    'SP500_tweets': ''
                }
                data.append(week_data)
                
            df = pd.DataFrame(data, columns=columns)
            print(f"✅ Created structure for {len(df)} weeks")
            return df
        
    def random_delay(self):
        """Random delay between 45-90 seconds"""
        delay = random.uniform(45, 90)
        print(f"⏳ Waiting {delay:.1f} seconds before next API call...")
        
        # Show countdown every 10 seconds
        for remaining in range(int(delay), 0, -10):
            time.sleep(min(10, remaining))
            if remaining > 10:
                print(f"   ⏰ {remaining} seconds remaining...")
        
        # Sleep any remaining time
        remaining_sleep = delay - int(delay)
        if remaining_sleep > 0:
            time.sleep(remaining_sleep)
            
    def search_tweets(self, query: str, asset: str) -> Optional[str]:
        """Search for tweets using Vela X API and return raw JSON string"""
        print(f"\n🔍 Searching {asset} tweets...")
        print(f"📝 Query: {query}")
        
        try:
            # URL encode the query properly
            encoded_query = urllib.parse.quote(query)
            full_url = f"{self.api_url}?query={encoded_query}"
            print(f"🌐 URL: {full_url}")
            
            response = requests.get(full_url)
            print(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                # Parse and format JSON nicely
                try:
                    data = response.json()
                    if isinstance(data, list):
                        tweet_count = len(data)
                    else:
                        tweet_count = len(data.get('tweets', [])) if isinstance(data, dict) else 0
                    print(f"✅ Found {tweet_count} {asset} tweets")
                    
                    # Format JSON with proper indentation
                    formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
                    return formatted_json
                    
                except Exception as e:
                    print(f"❌ Error parsing JSON: {e}")
                    # Fallback to raw text if JSON parsing fails
                    return response.text
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching {asset} tweets: {e}")
            return None
    
    def is_cell_empty(self, df: pd.DataFrame, week_idx: int, asset: str) -> bool:
        """Check if a specific cell (week + asset) is empty/missing"""
        col = f'{asset}_tweets'
        value = df.iloc[week_idx][col]
        # Cell is empty if it's NaN or empty string
        return pd.isna(value) or str(value).strip() == ''
    
    def collect_missing_tweets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill in only missing individual cells (week + asset combinations)"""
        if len(df) == 0:
            print("❌ No weeks to process")
            return df
            
        total_weeks = len(df)
        print(f"\n🐦 FILLING MISSING TWEET DATA")
        print(f"📊 Scanning {total_weeks} weeks × {len(self.assets)} assets = {total_weeks * len(self.assets)} cells")
        print(f"⏰ Random delays: 45-90 seconds between API calls")
        
        # First, scan all cells to find missing ones
        missing_cells = []
        for week_idx in range(total_weeks):
            for asset in self.assets:
                if self.is_cell_empty(df, week_idx, asset):
                    missing_cells.append((week_idx, asset))
        
        total_missing = len(missing_cells)
        if total_missing == 0:
            print("🎉 No missing data found! All cells are filled.")
            return df
            
        print(f"🔍 Found {total_missing} missing cells to fill")
        print(f"📈 This will take approximately {total_missing * 67.5 / 60:.1f} minutes")
        
        processed_cells = 0
        first_call = True
        
        # Process each missing cell individually
        for week_idx, asset in missing_cells:
            week_num = week_idx + 1
            start_date = df.iloc[week_idx]['start_date']
            end_date = df.iloc[week_idx]['end_date']
            
            processed_cells += 1
            
            print(f"\n" + "="*60)
            print(f"📅 CELL {processed_cells}/{total_missing}: Week {week_num} {asset}")
            print(f"🗓️  Date Range: {start_date} to {end_date}")
            
            # Add delay before each API call (except the very first)
            if not first_call:
                self.random_delay()
            first_call = False
            
            # Build search query with engagement minimum
            search_term = self.asset_queries[asset]
            query = f"{search_term} since:{start_date} until:{end_date} (min_faves:100 OR min_retweets:20)"
            
            # Search tweets
            raw_json = self.search_tweets(query, asset)
            
            # Store in dataframe (only update this specific cell)
            if raw_json:
                df.at[week_idx, f'{asset}_tweets'] = raw_json
                print(f"✅ {asset}: Stored raw JSON data")
            else:
                df.at[week_idx, f'{asset}_tweets'] = ''
                print(f"❌ {asset}: No data to store")
            
            # Save CSV after each API call for real-time updates
            print(f"💾 Updating CSV file...")
            df.to_csv('raw_tweets_data.csv', index=False)
            print(f"✅ CSV updated with Week {week_num} {asset} data")
            
        print(f"\n" + "="*60)
        print(f"📊 COLLECTION COMPLETE!")
        print(f"   ✅ Filled: {processed_cells} missing cells")
        print(f"   📈 Progress: {processed_cells}/{total_missing} cells processed")
        
        return df
    
    def save_csv(self, df: pd.DataFrame) -> bool:
        """Save the dataframe to CSV"""
        try:
            print(f"\n💾 Saving raw_tweets_data.csv...")
            df.to_csv('raw_tweets_data.csv', index=False)
            print(f"✅ Successfully saved {len(df)} weeks to raw_tweets_data.csv")
            
            # Show file size
            import os
            file_size = os.path.getsize('raw_tweets_data.csv')
            print(f"📁 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return False
    
    def show_summary(self, df: pd.DataFrame):
        """Show collection summary"""
        print(f"\n📋 FINAL COLLECTION SUMMARY:")
        print(f"   📁 Total weeks in CSV: {len(df)}")
        
        # Count populated weeks per asset
        asset_counts = {}
        for asset in self.assets:
            col = f'{asset}_tweets'
            if col in df.columns:
                non_empty = (df[col] != '').sum()
                asset_counts[asset] = non_empty
                
        # Overall populated weeks
        populated_weeks = max(asset_counts.values()) if asset_counts else 0
        print(f"   ✅ Weeks with tweet data: {populated_weeks}")
        
        # Per-asset breakdown
        print(f"\n📊 Data by Asset:")
        for asset in self.assets:
            count = asset_counts.get(asset, 0)
            print(f"   {asset}: {count}/{len(df)} weeks")
            
        # Show completion percentage
        completion_pct = (populated_weeks / len(df) * 100) if len(df) > 0 else 0
        print(f"\n📈 Completion: {completion_pct:.1f}% ({populated_weeks}/{len(df)} weeks)")
        
        # Show file size
        try:
            import os
            file_size = os.path.getsize('raw_tweets_data.csv')
            print(f"📁 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        except:
            pass

def main():
    print("🚀 STARTING MISSING TWEET COLLECTION")
    print("=" * 60)
    
    # Initialize collector
    collector = RawTweetCollector()
    
    # Load existing CSV (never recreate!)
    df = collector.load_existing_csv()
    
    # Fill only missing cells
    df = collector.collect_missing_tweets(df)
    
    # Show summary
    collector.show_summary(df)
    
    print(f"\n🎉 SUCCESS! Missing tweets filled in raw_tweets_data.csv")
    print(f"📝 Next steps: Process this data to extract top tweets for training")

if __name__ == "__main__":
    main() 