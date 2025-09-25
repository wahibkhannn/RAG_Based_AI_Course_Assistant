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
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None

transcripts = os.listdir("transcripts")

my_dicts = []
chunk_id = 0

for transcript in transcripts:
    with open(f"transcripts/{transcript}", "r", encoding="utf-8") as f:
      content = json.load(f)

    for chunk in content.get("chunks", []):
        # Handle case where chunk might be a string instead of a dict
        if isinstance(chunk, str):
            text_value = chunk
            chunk_data = {"text": text_value}
        else:
            text_value = chunk.get("text", "")
            chunk_data = chunk

        embedding = create_embedding(chunk["text"])

        chunk_data["chunk_id"] = chunk_id
        chunk_data["embedding"] = embedding 
        chunk_id+=1
        my_dicts.append(chunk_data)
        # print(chunk)
    break

print(json.dumps(my_dicts, indent=4))




# a = create_embedding("Cat sat on a mat")
# print(a)
