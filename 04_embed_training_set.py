#!/usr/bin/env python3
"""
04_embed_training_set.py

Creates market condition feature vectors for Long-Term Memory (LTM) similarity matching.
Instead of text embeddings, we create numerical vectors capturing market dynamics.

Process:
1. Load final_training_set.csv (first 40 training weeks)
2. Extract market features: 30-day changes, price levels, volatility, sentiment
3. Create feature vectors for each week
4. Build FAISS index for fast similarity search
5. Save index and metadata for LTM retrieval during test weeks 41-51

Market Features:
- SOL/BTC/ETH/SP500 30-day percentage changes
- Normalized SOL price level
- Market volatility indicators
- Cross-asset correlation patterns
- Tweet sentiment signals
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from typing import List, Dict, Any, Tuple
import faiss
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def load_training_data() -> pd.DataFrame:
    """Load the training dataset (first 40 weeks only)."""
    try:
        df = pd.read_csv("final_training_set.csv")
        # Only use first 40 weeks for training
        df_training = df.head(40).copy()
        print(f"✅ Loaded {len(df_training)} training examples (weeks 1-40)")
        return df_training
    except FileNotFoundError:
        print("❌ final_training_set.csv not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def extract_market_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extract market condition features for each week.
    
    Returns:
        - Feature matrix (n_weeks x n_features)
        - Feature names for interpretability
    """
    features = []
    feature_names = [
        'sol_30d_change', 'btc_30d_change', 'eth_30d_change', 'sp500_30d_change',
        'sol_price_normalized', 'market_volatility', 'cross_asset_correlation',
        'tweet_sentiment_score', 'market_stress_indicator', 'trend_strength'
    ]
    
    print(f"🔄 Extracting market features for {len(df)} weeks...")
    
    # Normalize SOL prices (0-1 scale based on training set range)
    sol_prices = df['prediction_date_sol_price'].values
    sol_min, sol_max = sol_prices.min(), sol_prices.max()
    
    for idx, row in df.iterrows():
        # Core 30-day changes
        sol_change = row['sol_30d_change_pct']
        btc_change = row['btc_30d_change_pct'] 
        eth_change = row['eth_30d_change_pct']
        sp_change = row['sp500_30d_change_pct']
        
        # Normalized SOL price level
        sol_price_norm = (row['prediction_date_sol_price'] - sol_min) / (sol_max - sol_min)
        
        # Market volatility indicator (based on change magnitudes)
        changes = [abs(sol_change), abs(btc_change), abs(eth_change), abs(sp_change)]
        market_volatility = np.mean(changes)
        
        # Cross-asset correlation pattern
        crypto_changes = [sol_change, btc_change, eth_change]
        correlation_strength = np.std(crypto_changes)  # Higher std = less correlated
        
        # Tweet sentiment score (based on tweet data length as proxy)
        tweet_data = str(row.get('tweets', ''))
        tweet_sentiment = min(len(tweet_data) / 2000.0, 1.0)  # Normalize to 0-1
        
        # Market stress indicator (extreme negative moves)
        stress_changes = [min(c, 0) for c in [sol_change, btc_change, eth_change]]
        market_stress = abs(np.mean(stress_changes))
        
        # Trend strength (consistency of direction)
        trend_directions = [1 if c > 0 else -1 for c in [sol_change, btc_change, eth_change]]
        trend_strength = abs(np.mean(trend_directions))
        
        # Combine into feature vector
        week_features = [
            sol_change, btc_change, eth_change, sp_change,
            sol_price_norm, market_volatility, correlation_strength,
            tweet_sentiment, market_stress, trend_strength
        ]
        
        features.append(week_features)
        
        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(df)} weeks")
    
    features_array = np.array(features, dtype=np.float32)
    print(f"✅ Created feature matrix shape: {features_array.shape}")
    
    return features_array, feature_names

def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features for better similarity matching.
    
    Returns:
        - Normalized features
        - Fitted scaler for future use
    """
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    print(f"✅ Normalized features - mean: {features_normalized.mean():.3f}, std: {features_normalized.std():.3f}")
    
    return features_normalized.astype(np.float32), scaler

def create_faiss_index(features: np.ndarray) -> faiss.Index:
    """
    Create a FAISS index for fast similarity search using cosine similarity.
    
    Args:
        features: Normalized feature matrix
    
    Returns:
        FAISS index
    """
    dimension = features.shape[1]
    
    # Use IndexFlatIP for inner product similarity (cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize features for cosine similarity
    faiss.normalize_L2(features)
    
    # Add features to index
    index.add(features)
    
    print(f"✅ Created FAISS index with {index.ntotal} vectors of dimension {dimension}")
    
    return index

def prepare_metadata(df: pd.DataFrame, features: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Prepare metadata for each training example including outcomes and reasoning.
    
    Args:
        df: Training dataframe
        features: Feature matrix
        feature_names: Names of features
    
    Returns:
        List of metadata dictionaries
    """
    metadata = []
    
    for idx, row in df.iterrows():
        # Create feature dict for interpretability
        feature_dict = {name: float(features[idx][i]) for i, name in enumerate(feature_names)}
        
        meta = {
            'week_index': idx + 1,  # 1-based week numbering
            'target_date': row['target_date'],
            'prediction_date': row['prediction_date'],
            'prediction_date_sol_price': float(row['prediction_date_sol_price']),
            'target_date_sol_price': float(row['target_date_sol_price']),
            'actual_change_pct': float(row['actual_change_pct']),
            'predicted_price': float(row['predicted_price']),
            'predicted_change_pct': float(row['predicted_change_pct']),
            'direction_correct': (row['actual_change_pct'] > 0) == (row['predicted_change_pct'] > 0),
            'price_error': abs(float(row['predicted_price']) - float(row['target_date_sol_price'])),
            'llm_reasoning': str(row['llm_reasoning'])[:500] + "..." if len(str(row['llm_reasoning'])) > 500 else str(row['llm_reasoning']),
            'llm_reflection': str(row['llm_reflection'])[:500] + "..." if len(str(row['llm_reflection'])) > 500 else str(row['llm_reflection']),
            'market_features': feature_dict,
            'tweets_summary': str(row.get('tweets', ''))[:200] + "..." if len(str(row.get('tweets', ''))) > 200 else str(row.get('tweets', ''))
        }
        metadata.append(meta)
    
    return metadata

def save_ltm_system(index: faiss.Index, metadata: List[Dict[str, Any]], scaler: StandardScaler, feature_names: List[str]):
    """
    Save the Long-Term Memory system components to disk.
    
    Args:
        index: FAISS index
        metadata: List of metadata dictionaries
        scaler: Fitted StandardScaler
        feature_names: List of feature names
    """
    os.makedirs("memory", exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, "memory/ltm_faiss_index.bin")
    print("✅ Saved FAISS index to memory/ltm_faiss_index.bin")
    
    # Save metadata
    with open("memory/ltm_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("✅ Saved metadata to memory/ltm_metadata.pkl")
    
    # Save scaler for normalizing future queries
    with open("memory/ltm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✅ Saved scaler to memory/ltm_scaler.pkl")
    
    # Save feature names for reference
    with open("memory/ltm_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print("✅ Saved feature names to memory/ltm_feature_names.json")

def test_similarity_search(index: faiss.Index, features: np.ndarray, metadata: List[Dict[str, Any]]):
    """
    Test the similarity search functionality with sample queries.
    """
    print("\n🧪 Testing LTM similarity search...")
    
    # Test Case 1: Find similar market conditions to Week 15
    test_week_idx = 14  # Week 15 (0-indexed)
    test_query = features[test_week_idx:test_week_idx+1]  # Shape (1, dimension)
    
    # Search for top 3 most similar examples (excluding itself)
    k = 4  # Get 4 to exclude the exact match
    scores, indices = index.search(test_query, k)
    
    query_meta = metadata[test_week_idx]
    print(f"\n📝 Query Week {query_meta['week_index']} ({query_meta['target_date']}):")
    print(f"   Market: SOL {query_meta['market_features']['sol_30d_change']:.1f}%, BTC {query_meta['market_features']['btc_30d_change']:.1f}%")
    print(f"   Outcome: Predicted {query_meta['predicted_change_pct']:.1f}%, Actual {query_meta['actual_change_pct']:.1f}%")
    print(f"   Direction: {'✅' if query_meta['direction_correct'] else '❌'}")
    
    print(f"\n🔍 Top 3 similar market conditions:")
    
    similar_count = 0
    for score, idx in zip(scores[0], indices[0]):
        if idx != test_week_idx and similar_count < 3:  # Exclude exact match
            similar_meta = metadata[idx]
            print(f"   {similar_count + 1}. Week {similar_meta['week_index']} (similarity: {score:.3f})")
            print(f"      Market: SOL {similar_meta['market_features']['sol_30d_change']:.1f}%, BTC {similar_meta['market_features']['btc_30d_change']:.1f}%")
            print(f"      Outcome: Predicted {similar_meta['predicted_change_pct']:.1f}%, Actual {similar_meta['actual_change_pct']:.1f}%")
            print(f"      Direction: {'✅' if similar_meta['direction_correct'] else '❌'}")
            print()
            similar_count += 1

def main():
    """Main function to create the LTM embedding system."""
    print("🚀 Creating Long-Term Memory (LTM) system for market condition similarity...")
    
    # Load training data (first 40 weeks)
    df = load_training_data()
    if df is None:
        return
    
    # Check for required columns
    required_cols = ['sol_30d_change_pct', 'btc_30d_change_pct', 'eth_30d_change_pct', 'sp500_30d_change_pct', 
                     'prediction_date_sol_price', 'predicted_price', 'predicted_change_pct', 'actual_change_pct']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return
    
    # Extract market features
    features, feature_names = extract_market_features(df)
    
    # Normalize features
    features_normalized, scaler = normalize_features(features)
    
    # Create FAISS index
    index = create_faiss_index(features_normalized)
    
    # Prepare metadata
    metadata = prepare_metadata(df, features, feature_names)
    
    # Save LTM system
    save_ltm_system(index, metadata, scaler, feature_names)
    
    # Test the system
    test_similarity_search(index, features_normalized, metadata)
    
    print("\n🎉 Long-Term Memory (LTM) system created successfully!")
    print("\n📋 System Components:")
    print("   • memory/ltm_faiss_index.bin - Fast similarity search index")
    print("   • memory/ltm_metadata.pkl - Training examples with outcomes")
    print("   • memory/ltm_scaler.pkl - Feature normalization")
    print("   • memory/ltm_feature_names.json - Feature definitions")
    print("\n🎯 Next Steps:")
    print("   • Use this system in 05_predict_with_memory.py")
    print("   • For test weeks 41-51: find 3 most similar historical examples")
    print("   • Combine with STM (3 recent weeks) for enhanced predictions")

if __name__ == "__main__":
    main() 