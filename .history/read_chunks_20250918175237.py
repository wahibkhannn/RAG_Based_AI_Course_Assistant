import requests

requests.post("http://localhost:11434/api/embeddings", json={
    "model": "bge-m3",
    "prompt": "I am using my laptop currently to write code"})