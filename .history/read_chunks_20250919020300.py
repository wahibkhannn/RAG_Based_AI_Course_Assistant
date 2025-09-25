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
            time.sleep(0.1)
            

    break


if my_dicts:
    df = pd.DataFrame.from_records(my_dicts)
    print(f"DataFrame shape: {df.shape}")
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
else:
    print("No data to create DataFrame - all embedding requests failed")

incoming_query = input("Ask a question:")
query_embedding = create_embedding(incoming_query)[0]
print(f"Query embedding length: {len(query_embedding)}")

# 
