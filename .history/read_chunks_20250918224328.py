import requests
import os
def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "prompt": text
        })
    embedding = r.json()['embedding']
    return embedding

transcripts = os.listdir("transcripts")
for transcript in 
# a = create_embedding("Cat sat on a mat")
# print(a)
