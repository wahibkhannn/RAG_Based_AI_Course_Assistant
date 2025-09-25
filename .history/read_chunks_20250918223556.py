import requests

def create_e
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "prompt": "I am using my laptop currently to write code"
        })

embedding = r.json()['embedding']
print(embedding[0:8])
