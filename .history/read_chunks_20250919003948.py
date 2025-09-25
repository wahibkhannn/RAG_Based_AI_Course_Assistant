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


import requests
import os 
import json 
def create_embedding(text): 
        try: 
            r = requests.post("http://localhost:11434/api/embeddings", json={
             "model": "bge-m3",
               "input": text }) 
             r.raise_for_status() # Raise an error if request failed embedding = r.json()['embedding'] return embedding except requests.exceptions.RequestException as e: print(f"Request error: {e}") return None transcripts = os.listdir("transcripts") my_dicts = [] chunk_id = 0 for transcript in transcripts: with open(f"transcripts/{transcript}", "r", encoding="utf-8") as f: content = json.load(f) for chunk in content.get("chunks", []): embedding = create_embedding(chunk["text"]) chunk["chunk_id"] = chunk_id chunk["embedding"] = embedding chunk_id+=1 my_dicts.append(chunk) # print(chunk) break print(my_dicts)