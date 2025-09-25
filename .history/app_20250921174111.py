# app.py
import streamlit as st
import pandas as pd
import json
from process_incoming_query import create_embedding, inference, df, LLM_MODEL
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Java Course AI Assistant", layout="wide")

st.title("📚 Java Course AI Assistant")

# User query
user_query = st.text_input("Ask a question about the Java course:")

if st.button("Get Answer") and user_query.strip() != "":
    st.info("Processing your query...")
    
    # Create embedding
    query_embedding = create_embedding(user_query)
    
    # Compute similarities
    similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding]).flatten()
    df['similarity'] = similarities
    df_sorted = df.sort_values(by='similarity', ascending=False).head(10)
    
    # Build context
    context_json = df_sorted[['title', 'text', 'start', 'end', 'similarity']].to_dict(orient='records')
    
    # Build prompt
    

    # Get AI response
    answer = inference(prompt, model=LLM_MODEL)
    
    # Display
    st.markdown("### 📝 Answer")
    st.write(answer)
    
    st.markdown("### 🎬 Top Relevant Video Chunks")
    st.dataframe(df_sorted[['title', 'text', 'start', 'end', 'similarity']])
