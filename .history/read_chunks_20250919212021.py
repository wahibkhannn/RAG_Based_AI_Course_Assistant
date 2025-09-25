import pandas as pd
import requests
import os
import numpy as np
import time
import json 
import joblib



def create_embedding(text): 
    try: 
        r = requests.post("http://localhost:11434/api/embeddings", json={
            "model": "bge-m3",
            "prompt": text }) 
        r.raise_for_status() # Raise an error if request failed
        response_data = r.json()
        # print(f"API Response keys: {response_data.keys()}") 

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
    
    title = os.path.splitext(transcript)[0]
    number = os.path.splitext(transcript)[0].split(" ")[0]
    print(f"Processing: {title}")

    with open(f"transcripts/{transcript}") as f: 
        content = json.load(f)
        print(f"Loaded {len(content['chunks'])} chunks from {transcript}")

    # embeddings = create_embedding(content['chunks'])

    for i, chunk in enumerate(content['chunks']): 
  
        embedding = create_embedding(chunk['text'])
        if embedding:
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embedding
            chunk["title"] = title
            chunk["number"] = number

            my_dicts.append(chunk)
            chunk_id += 1
            time.sleep(0.01)
    


if my_dicts:
    df = pd.DataFrame.from_records(my_dicts)
    print(f"DataFrame shape: {df.shape}")
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
    joblib.dump(df, "embeddings_df.joblib")
else:
    print("No data to create DataFrame - all embedding requests failed")

