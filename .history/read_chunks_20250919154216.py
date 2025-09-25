# import requests

# def create_embedding(text):
#     r = requests.post("http://localhost:11434/api/embeddings", json={
#         "model": "bge-m3",
#         "prompt": text
#         })
#     embedding = r.json()['embedding']
#     return embedding


# a = create_embedding("Hello, world!")
# print(a)

import pandas as pd
import requests
import os 
import time
import json 

from sklearn.metrics.pairwise import cosine_similarity
def create_embedding(text): 
    try: 
        r = requests.post("http://localhost:11434/api/embeddings", json={
            "model": "bge-m3",
            "prompt": text }) 
        r.raise_for_status() # Raise an error if request failed
        response_data = r.json()
        print(f"API Response keys: {response_data.keys()}") 

        embedding = r.json()['embedding'] 
        return embedding 
    except requests.exceptions.RequestException as e:
         print(f"Request error: {e}") 
         return None
    except json.JSONDecodeError as e:
         print(f"JSON decode error: {e}")
         return None
    except Exception as e:
         print(f"Unexpected error: {e}")
         return None 
    
transcripts = os.listdir("transcripts") 

my_dicts = [] 
chunk_id = 0 

for transcript in transcripts:
    print(f"Processing: {transcript}")

    with open(f"transcripts/{transcript}") as f: 
        content = json.load(f)
        print(f"Loaded {len(content['chunks'])} chunks from {transcript}")

    embeddings = create_embedding(content['chunks'])

    for i, chunk in enumerate(content['chunks']): 
  
        embedding = create_embedding(chunk['text'])
        if embedding:
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embedding
            my_dicts.append(chunk)
            chunk_id += 1
            time.sleep(0.01)
        if i==5:
         break

    break


if my_dicts:
    df = pd.DataFrame.from_records(my_dicts)
    print(f"DataFrame shape: {df.shape}")
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
else:
    print("No data to create DataFrame - all embedding requests failed")

incoming_query = input("Ask a question:")
query_embedding = create_embedding(incoming_query)
print(f"Query embedding length: {len(query_embedding)}")

# Compute cosine similarities
# Find similarities of query_embedding with all chunk embeddings
print(df['embedding'].values[0][:7])
print(df['embedding'].shape)
# similarities = cosine_similarity([query_embedding], df['embedding'].tolist())[0]
similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding])

# Attach similarity scores to the dataframe
df['similarity'] = similarities
print(similarities)
# Sort by highest similarity
df_sorted = df.sort_values(by='similarity', ascending=False)

print("\nTop 5 most similar chunks:")
print(df_sorted[['chunk_id', 'text', 'similarity']].head())