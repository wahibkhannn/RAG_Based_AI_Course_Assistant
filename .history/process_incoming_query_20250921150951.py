import pandas as pd
import requests
import numpy as np 
import joblib
import json
# from read_chunks import create_embedding
from sklearn.metrics.pairwise import cosine_similarity


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
     r= requests.post("http://localhost:11")

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
df_sorted = df.sort_values(by='similarity', ascending=False)
max_indx = similarities.argsort()[-7:][::-1]

prompt = f'''I am teaching Java course. Here are video chunks containing video title with their number , the text, start time 
and end time in seconds, the text at that time :

{df_sorted[['title', 'text', 'similarity', 'start', 'end']].to_json()}
------------------------------------
{incoming_query}
User asked this question related to the video chunks, you have to answer where and how much
content is taught where in which video at what timestamp and guide the user to go to that particular video. 
If user asks unrelated question, tell him that you can only answer questions related to the course. 
'''

print("Top 7 most similar chunk indices:", max_indx)
print("-------------------------------------")
print("\nTop 15 most similar chunks:")
print(df_sorted[['title', 'text', 'similarity', 'start', 'end']].head(15))
print("-------------------------------------")


