#!/usr/bin/env python3
"""
04_embed_training_set.py

Creates embeddings of training examples and stores them in a FAISS vector database
for fast similarity retrieval. This enables the Long-Term Memory (LTM) system.

Process:
1. Load training_set.csv
2. Extract summarized_context for each week
3. Generate embeddings using OpenAI's text-embedding-3-small
4. Build FAISS index for similarity search
5. Save index and metadata for later retrieval
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
import faiss
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_training_data() -> pd.DataFrame:
    """Load the training dataset."""
    try:
        df = pd.read_csv("training_set.csv")
        print(f"✅ Loaded {len(df)} training examples")
        return df
    except FileNotFoundError:
        print("❌ training_set.csv not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def generate_embeddings(texts: List[str], batch_size: int = 10) -> np.ndarray:
    """
    Generate embeddings for a list of texts using OpenAI's embedding model.
    
    Args:
        texts: List of strings to embed
        batch_size: Number of texts to process at once
    
    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    all_embeddings = []
    
    print(f"🔄 Generating embeddings for {len(texts)} examples...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [embedding.embedding for embedding in response.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"   Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
        except Exception as e:
            print(f"❌ Error generating embeddings for batch {i//batch_size + 1}: {e}")
            # Create zero embeddings as fallback
            fallback_embeddings = [[0.0] * 1536] * len(batch)  # text-embedding-3-small has 1536 dimensions
            all_embeddings.extend(fallback_embeddings)
    
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f"✅ Generated embeddings shape: {embeddings_array.shape}")
    
    return embeddings_array

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Create a FAISS index for fast similarity search.
    
    Args:
        embeddings: numpy array of embeddings
    
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP for inner product similarity (cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings)
    
    print(f"✅ Created FAISS index with {index.ntotal} vectors")
    
    return index

def prepare_metadata(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Prepare metadata for each training example.
    
    Args:
        df: Training dataframe
    
    Returns:
        List of metadata dictionaries
    """
    metadata = []
    
    for idx, row in df.iterrows():
        meta = {
            'week_index': idx,
            'target_date': row['target_date'],
            'context_start': row['context_start'],
            'target_price': row['target_price'],
            'prediction_end': row['prediction_end'],
            'actual_price': row['actual_price'],
            'predicted_price': row['predicted_price'],
            'actual_change_pct': row['actual_change_pct'],
            'predicted_change_pct': row['predicted_change_pct'],
            'llm_reasoning': row['llm_reasoning'],
            'llm_reflection': row['llm_reflection'],
            'summarized_context': row['summarized_context']
        }
        metadata.append(meta)
    
    return metadata

def save_embeddings_system(index: faiss.Index, metadata: List[Dict[str, Any]]):
    """
    Save the FAISS index and metadata to disk.
    
    Args:
        index: FAISS index
        metadata: List of metadata dictionaries
    """
    os.makedirs("memory", exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, "memory/faiss_index.bin")
    print("✅ Saved FAISS index to memory/faiss_index.bin")
    
    # Save metadata
    with open("memory/embeddings.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("✅ Saved metadata to memory/embeddings.pkl")

def test_similarity_search(index: faiss.Index, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
    """
    Test the similarity search functionality with a sample query.
    
    Args:
        index: FAISS index
        embeddings: Original embeddings array
        metadata: Metadata list
    """
    print("\n🧪 Testing similarity search...")
    
    # Use the first example as a test query
    test_query = embeddings[0:1]  # Shape (1, dimension)
    
    # Search for top 3 most similar examples
    k = 3
    scores, indices = index.search(test_query, k)
    
    query_context = metadata[0]['summarized_context'][:100] + "..."
    print(f"\n📝 Query context: {query_context}")
    print(f"\n🔍 Top {k} similar examples:")
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        similar_context = metadata[idx]['summarized_context'][:100] + "..."
        date = metadata[idx]['target_date']
        print(f"   {i+1}. Week {date} (score: {score:.3f})")
        print(f"      Context: {similar_context}")
        print()

def main():
    """Main function to create the embedding system."""
    print("🚀 Creating embedding system for inference-time memory...")
    
    # Load training data
    df = load_training_data()
    if df is None:
        return
    
    # Filter out rows with missing context (if any)
    df_clean = df.dropna(subset=['summarized_context'])
    if len(df_clean) < len(df):
        print(f"⚠️  Filtered out {len(df) - len(df_clean)} rows with missing context")
    
    # Extract text contexts for embedding
    contexts = df_clean['summarized_context'].tolist()
    print(f"📝 Preparing to embed {len(contexts)} context summaries")
    
    # Generate embeddings
    embeddings = generate_embeddings(contexts)
    
    # Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Prepare metadata
    metadata = prepare_metadata(df_clean)
    
    # Save everything
    save_embeddings_system(index, metadata)
    
    # Test the system
    test_similarity_search(index, embeddings, metadata)
    
    print("\n🎉 Embedding system created successfully!")
    print("\nNext steps:")
    print("- Phase 5: Use this system in 05_predict_with_memory.py")
    print("- The system will find similar past situations for better predictions")

if __name__ == "__main__":
    main() 