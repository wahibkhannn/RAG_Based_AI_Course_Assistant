import pandas as pd
import requests
from datetime import datetime
import numpy as np 
import joblib
import json
import os
# from read_chunks import create_embedding
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("logs", exist_ok=True)

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

print("Loading the embeddings!!")
df = joblib.load("embeddings_df.joblib")


def inference(prompt, model):
    r= requests.post("http://localhost:11434/api/generate", json={
          "model": "deepseek-r1",
          "prompt": prompt,
          "stream": False

    })
    # response = r.json()
    # print(response)
    r.raise_for_status()
    data = r.json()

    # Extract clean text from response
    output_text = data.get("response", "").strip()

    # Save raw JSON for debugging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"logs/raw_response_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

        return output_text
    

incoming_query = input("Ask a question:")
query_embedding = create_embedding(incoming_query)

print(f"Query embedding length: {len(query_embedding)}")

# Compute cosine similarities
# Find similarities of query_embedding with all chunk embeddings
# print(df['embedding'].values[0][:7])
print(df['embedding'].shape)

# similarities = cosine_similarity([query_embedding], df['embedding'].tolist())[0]
similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding]).flatten()

# Attach similarity scores to the dataframe
df['similarity'] = similarities
# print(similarities)
# Sort by highest similarity
df_sorted = df.sort_values(by='similarity', ascending=False).head(10)
max_indx = similarities.argsort()[-7:][::-1]


# Build Context 
context_json = df_sorted[['title', 'text', 'start', 'end', 'similarity']].to_dict(orient='records')

# Save top chunks to file for review
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
context_file = f"logs/context_{timestamp}.json"
with open(context_file, "w", encoding="utf-8") as f:
    json.dump(context_json, f, ensure_ascii=False, indent=4)


print(f"\n[INFO] Top {T} relevant chunks saved to {context_file}\n")

prompt = f'''I am teaching Java course. Here are video chunks containing video title with their number , the text, start time 
and end time in seconds, the text at that time :

{df_sorted[['title', 'text', 'similarity', 'start', 'end']].to_json()}
------------------------------------
User Question: {incoming_query}

'''

prompt_file = f"logs/prompt_{timestamp}.txt"
with open(prompt_file, "w", encoding="utf-8") as f:
    f.write(prompt)
print(f"[INFO] Prompt saved to {prompt_file}")


# RUN INFERENCE
# ======================
answer = inference(prompt)
print("\n=== AI Response ===\n")
print(answer)
print("\n===================\n")




print("-------------------------------------")
print("\nTop 15 most similar chunks:")
print(df_sorted[['title', 'text', 'similarity', 'start', 'end']].head(15))
print("-------------------------------------")


