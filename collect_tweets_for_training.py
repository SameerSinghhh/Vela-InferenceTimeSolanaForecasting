#!/usr/bin/env python3
"""
Collect tweets for training dataset using Vela X API.
Focus on first week only for testing.
"""

import pandas as pd
import requests
import time
import random
import json
from datetime import datetime
from typing import List, Dict, Optional

class TweetCollector:
    def __init__(self):
        self.api_url = "https://x.vela.partners/search"
        self.assets = {
            'BTC': '(bitcoin OR $BTC OR #Bitcoin)',
            'ETH': '(ethereum OR $ETH OR #Ethereum)', 
            'SOL': '(solana OR $SOL OR #Solana)',
            'SP500': '(S&P500 OR #SP500 OR "S&P 500")'
        }
        
    def random_delay(self):
        """Random delay between 45-75 seconds"""
        delay = random.uniform(45, 75)
        print(f"⏳ Waiting {delay:.1f} seconds before next API call...")
        time.sleep(delay)
        
    def search_tweets(self, query: str, asset: str) -> Optional[List[Dict]]:
        """Search for tweets using Vela X API"""
        print(f"\n🔍 Searching {asset} tweets...")
        print(f"📝 Query: {query}")
        
        try:
            response = requests.get(self.api_url, params={'q': query})
            print(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                tweets = data.get('tweets', []) if isinstance(data, dict) else []
                print(f"✅ Found {len(tweets)} {asset} tweets")
                return tweets
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching {asset} tweets: {e}")
            return None
    
    def filter_and_rank_tweets(self, tweets: List[Dict], asset: str) -> List[Dict]:
        """Filter tweets by date range and rank by engagement"""
        if not tweets:
            return []
            
        print(f"📊 Processing {asset} tweets...")
        
        # Filter valid tweets with required fields
        valid_tweets = []
        for tweet in tweets:
            if all(key in tweet for key in ['views', 'likes', 'retweets', 'tweet_date']):
                # Calculate engagement score
                engagement = tweet.get('views', 0) + tweet.get('likes', 0) * 2 + tweet.get('retweets', 0) * 3
                tweet['engagement_score'] = engagement
                valid_tweets.append(tweet)
        
        # Sort by engagement score
        ranked_tweets = sorted(valid_tweets, key=lambda x: x['engagement_score'], reverse=True)
        
        print(f"📈 {len(ranked_tweets)} valid {asset} tweets ranked by engagement")
        if ranked_tweets:
            top_tweet = ranked_tweets[0]
            print(f"🏆 Top {asset} tweet: {top_tweet['engagement_score']} engagement score")
            
        return ranked_tweets[:3]  # Top 3
    
    def collect_tweets_for_period(self, context_start: str, prediction_date: str) -> Dict[str, List[Dict]]:
        """Collect tweets for all assets in a specific period"""
        print(f"\n🐦 COLLECTING TWEETS FOR PERIOD")
        print(f"📅 Date Range: {context_start} to {prediction_date} (exclusive)")
        
        all_tweets = {}
        
        for i, (asset, search_term) in enumerate(self.assets.items()):
            # Add delay before each API call (except the first)
            if i > 0:
                self.random_delay()
            
            # Build search query
            query = f"{search_term} since:{context_start} until:{prediction_date} min_faves:100"
            
            # Search tweets
            raw_tweets = self.search_tweets(query, asset)
            
            # Filter and rank
            top_tweets = self.filter_and_rank_tweets(raw_tweets, asset)
            all_tweets[asset] = top_tweets
            
            print(f"✅ {asset}: Selected {len(top_tweets)} top tweets")
        
        return all_tweets
    
    def format_tweets_for_csv(self, all_tweets: Dict[str, List[Dict]]) -> str:
        """Format tweets for CSV insertion"""
        formatted_tweets = []
        
        for asset, tweets in all_tweets.items():
            for i, tweet in enumerate(tweets, 1):
                tweet_text = tweet.get('text', '').replace('"', '""')  # Escape quotes
                formatted_tweet = f"{asset}_{i}: {tweet_text[:200]}..."  # Truncate long tweets
                formatted_tweets.append(formatted_tweet)
        
        return " | ".join(formatted_tweets)
    
    def update_training_data(self, tweets_text: str) -> bool:
        """Update the first row of final_training_set.csv with tweets"""
        try:
            print(f"\n💾 Updating final_training_set.csv...")
            
            # Load CSV
            df = pd.read_csv('final_training_set.csv')
            print(f"📊 Loaded {len(df)} rows")
            
            # Update first row tweets column
            df.at[0, 'tweets'] = tweets_text
            
            # Save back to CSV
            df.to_csv('final_training_set.csv', index=False)
            print(f"✅ Successfully updated Period 1 tweets in CSV")
            return True
            
        except Exception as e:
            print(f"❌ Error updating CSV: {e}")
            return False

def main():
    print("🚀 STARTING TWEET COLLECTION FOR TRAINING DATA")
    print("=" * 60)
    
    # Initialize collector
    collector = TweetCollector()
    
    # Load training data to get first period dates
    try:
        df = pd.read_csv('final_training_set.csv')
        first_row = df.iloc[0]
        
        context_start = first_row['context_start_date']
        prediction_date = first_row['prediction_date']
        
        print(f"🎯 PERIOD 1 TARGET:")
        print(f"   Context Start: {context_start}")
        print(f"   Prediction Date: {prediction_date}")
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return
    
    # Collect tweets for the first period
    all_tweets = collector.collect_tweets_for_period(context_start, prediction_date)
    
    # Show summary
    print(f"\n📋 COLLECTION SUMMARY:")
    total_tweets = sum(len(tweets) for tweets in all_tweets.values())
    print(f"   Total tweets collected: {total_tweets}")
    for asset, tweets in all_tweets.items():
        print(f"   {asset}: {len(tweets)} tweets")
    
    if total_tweets > 0:
        # Format for CSV
        tweets_text = collector.format_tweets_for_csv(all_tweets)
        print(f"\n📝 Formatted tweets length: {len(tweets_text)} characters")
        
        # Update CSV
        success = collector.update_training_data(tweets_text)
        
        if success:
            print(f"\n🎉 SUCCESS! Period 1 tweets added to final_training_set.csv")
        else:
            print(f"\n❌ Failed to update CSV file")
    else:
        print(f"\n⚠️ No tweets collected - skipping CSV update")

if __name__ == "__main__":
    main() 