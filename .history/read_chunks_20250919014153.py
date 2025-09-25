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
import json 
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

    embeddings = create_embedding(content[''])

    # for i, chunk in enumerate(content['chunks']): 
    #         chunk["chunk_id"] = chunk_id 
    #         chunk['embedding'] = embeddings[i]
    #         chunk_id+=1 
    #         my_dicts.append(chunk)
            # print(chunk) 
    for chunk in content['chunks']:
        embedding = create_embedding(chunk['text'])
        if embedding:
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embedding
            my_dicts.append(chunk)
            chunk_id += 1
            

    break  # remove this to process all files
# print(my_dicts)


df = pd.DataFrame(my_dicts)
print(df.head())