import pandas as pd
import requests
import os 
import json 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Optional

# Thread-safe session for connection pooling
session = requests.Session()
session.headers.update({'Content-Type': 'application/json'})

# Lock for thread-safe operations
lock = threading.Lock()

def create_embedding(text: str) -> Optional[List[float]]: 
    """Create embedding for a single text"""
    try: 
        r = session.post("http://localhost:11434/api/embeddings", json={
            "model": "bge-m3",
            "prompt": text
        }, timeout=30)
        r.raise_for_status()
        
        response_data = r.json()
        return response_data['embedding']
        
    except Exception as e:
        with lock:
            print(f"Error creating embedding: {e}")
        return None

def create_batch_embeddings(texts: List[str], max_workers: int = 5) -> List[Optional[List[float]]]:
    """Create embeddings for multiple texts using threading"""
    embeddings = [None] * len(texts)
    
    def process_text(index_text_pair):
        index, text = index_text_pair
        embedding = create_embedding(text)
        return index, embedding
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_text, (i, text)): i 
            for i, text in enumerate(texts)
        }
        
        # Collect results
        for future in as_completed(futures):
            try:
                index, embedding = future.result()
                embeddings[index] = embedding
                
                # Progress indicator
                completed = sum(1 for emb in embeddings if emb is not None)
                with lock:
                    print(f"Progress: {completed}/{len(texts)} embeddings created", end='\r')
                    
            except Exception as e:
                with lock:
                    print(f"Error processing batch item: {e}")
    
    print()  # New line after progress indicator
    return embeddings

def process_transcript_optimized(transcript_path: str) -> List[dict]:
    """Process a single transcript file efficiently"""
    print(f"Processing: {os.path.basename(transcript_path)}")
    
    with open(transcript_path) as f: 
        content = json.load(f)
    
    chunks = content['chunks']
    print(f"Loaded {len(chunks)} chunks")
    
    # Extract all texts for batch processing
    texts = [chunk['text'] for chunk in chunks]
    
    # Create embeddings in batches
    print("Creating embeddings...")
    embeddings = create_batch_embeddings(texts)
    
    # Prepare results
    processed_chunks = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if embedding:
            chunk_copy = chunk.copy()
            chunk_copy["chunk_id"] = i
            chunk_copy["embedding"] = embedding
            processed_chunks.append(chunk_copy)
        else:
            print(f"Failed to create embedding for chunk {i}")
    
    return processed_chunks

def find_similar_chunks(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Find most similar chunks to query"""
    print("Creating query embedding...")
    query_embedding = create_embedding(query)
    
    if not query_embedding:
        print("Failed to create query embedding")
        return pd.DataFrame()
    
    print("Computing similarities...")
    # Convert embeddings to numpy array for faster computation
    embeddings_matrix = np.array(df['embedding'].tolist())
    query_array = np.array([query_embedding])
    
    # Compute cosine similarities
    similarities = cosine_similarity(query_array, embeddings_matrix)[0]
    
    # Add similarities to dataframe
    df_with_sim = df.copy()
    df_with_sim['similarity'] = similarities
    
    # Return top k results
    return df_with_sim.sort_values(by='similarity', ascending=False).head(top_k)

# Main processing
def main():
    transcripts_dir = "transcripts"
    transcripts = [f for f in os.listdir(transcripts_dir) if f.endswith('.json')]
    
    if not transcripts:
        print("No JSON files found in transcripts directory")
        return
    
    all_chunks = []
    
    # Process each transcript (you can remove the break to process all)
    for transcript in transcripts[:1]:  # Process only first file for now
        transcript_path = os.path.join(transcripts_dir, transcript)
        chunks = process_transcript_optimized(transcript_path)
        all_chunks.extend(chunks)
        break  # Remove to process all files
    
    if not all_chunks:
        print("No chunks processed successfully")
        return
    
    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame(all_chunks)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Interactive query loop
    while True:
        try:
            query = input("\nAsk a question (or 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Find similar chunks
            results = find_similar_chunks(query, df, top_k=5)
            
            if not results.empty:
                print("\nTop 5 most similar chunks:")
                print("-" * 80)
                for _, row in results.iterrows():
                    print(f"Similarity: {row['similarity']:.4f}")
                    print(f"Text: {row['text'][:200]}...")
                    print(f"Time: {row['start']:.2f}s - {row['end']:.2f}s")
                    print("-" * 80)
            else:
                print("No similar chunks found")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()