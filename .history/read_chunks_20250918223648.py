import requests

def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "prompt": text
        })
    return em

embedding = r.json()['embedding']
print(embedding[0:8])
