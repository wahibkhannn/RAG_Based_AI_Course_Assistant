import requests
import os
import json 

def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "input": text
        })
    embedding = r.json()['embedding']
    return embedding

transcripts = os.listdir("transcripts")

my_dicts = []
for transcript in transcripts:
    with open(f"transcripts/{transcript}") as f:
      content = json.load(f)
    for chunk in content["chunks"]:
        print(chunk)
        
    break




# a = create_embedding("Cat sat on a mat")
# print(a)
