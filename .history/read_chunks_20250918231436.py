import requests
import os
import json 

def create_embedding(text):
    try:
        r = requests.post("http://localhost:11434/api/embeddings", json={
            "model": "bge-m3",
            "input": text
            })
        r.raise_for_status() # Raise an error if request failed
        embedding = r.json()['embedding']
        return embedding

transcripts = os.listdir("transcripts")

my_dicts = []
chunk_id = 0
for transcript in transcripts:
    with open(f"transcripts/{transcript}") as f:
      content = json.load(f)
    for chunk in content["chunks"]:
        
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = create_embedding(chunk["text"])
        chunk_id+=1
        my_dicts.append(chunk)
        # print(chunk)
    break

print(my_dicts)




# a = create_embedding("Cat sat on a mat")
# print(a)
