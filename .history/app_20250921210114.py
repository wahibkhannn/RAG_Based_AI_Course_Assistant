# app.py
import streamlit as st
import pandas as pd
import json
from process_incoming_query import create_embedding, inference, df
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Java Course AI Assistant", layout="wide")

st.title("📚 Java Course AI Assistant")

# --- User query ---
user_query = st.text_input("Ask a question about the Java course:")

if st.button("Get Answer") and user_query.strip() != "":
    st.info("Processing your query...")

    # --- Create query embedding ---
    query_embedding = create_embedding(user_query)
    if query_embedding is None:
        st.error("Failed to generate embedding for your query. Please try again.")
        st.stop()

    # --- Compute similarities ---
    similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding]).flatten()
    df['similarity'] = similarities

    # --- Get top 3 chunks (sorted) ---
    df_sorted = df.sort_values(by='similarity', ascending=False).head(3)

    # --- Truncate text for efficiency ---
    df_sorted['short_text'] = df_sorted['text'].str.slice(0, 200)

    # Build lightweight context
    chunks = [
        {
            "title": row["title"],
            "start": float(row["start"]),
            "end": float(row["end"]),
            "text": row["short_text"]
        }
        for _, row in df_sorted.iterrows()
    ]

    # --- Build optimized prompt ---
    prompt = f"""
You are an AI teaching assistant for a Java programming course.
ONLY use the following video chunks to answer.

If the question is unrelated to the course, respond exactly:
{{"answer": "I can only answer questions about this Java course.", "references": []}}

Video Chunks:
{json.dumps(chunks, indent=2)}

User Question: {user_query}

Respond ONLY in valid JSON like this:
{{
  "answer": "<concise answer>",
  "references": [
    {{
      "title": "<video title>",
      "start": <start_time>,
      "end": <end_time>
    }}
  ]
}}
"""

    # --- Get AI response ---
    answer = inference(prompt, model="deepseek-r1")

    # --- Display ---
    st.markdown("### 📝 Answer")
    try:
        parsed_answer = json.loads(answer)
        st.json(parsed_answer)
    except:
        st.write(answer)

    st.markdown("### 🎬 Top Relevant Video Chunks")
    st.dataframe(df_sorted[['title', 'short_text', 'start', 'end', 'similarity']])
