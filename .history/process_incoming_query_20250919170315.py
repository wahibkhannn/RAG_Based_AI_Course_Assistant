import pandas as pd
import requests
import numpy as np 
from read_chunks import create_embedding

incoming_query = input("Ask a question:")
query_embedding = create_embedding(incoming_query)
print(f"Query embedding length: {len(query_embedding)}")

# Compute cosine similarities
# Find similarities of query_embedding with all chunk embeddings
print(df['embedding'].values[0][:7])
print(df['embedding'].shape)
# similarities = cosine_similarity([query_embedding], df['embedding'].tolist())[0]
similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding]).flatten()

# Attach similarity scores to the dataframe
df['similarity'] = similarities
print(similarities)
# Sort by highest similarity
df_sorted = df.sort_values(by='similarity', ascending=False)

max_indx = similarities.argsort()[-5:][::-1]
print("Top 5 most similar chunk indices:", max_indx)
print("-------------------------------------")

print("\nTop 5 most similar chunks:")
print(df_sorted[['chunk_id', 'text', 'similarity']].head())