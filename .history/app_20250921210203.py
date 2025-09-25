# app.py
import streamlit as st
import pandas as pd
import json
from process_incoming_query import create_embedding, inference, df
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
    df_sorted = df.sort_values(by='similarity', ascending=False).head(5)
    
    # Build context
    context_json = df_sorted[['title', 'text', 'start', 'end', 'similarity']].to_dict(orient='records')
    
    # Build prompt
    # prompt = f'''You are an AI teaching assistant for a **Java programming course**. 
    # The course content is divided into video chunks, each with:
    # - A title that includes the video number and name,
    # - A timestamp (start and end in seconds),
    # - A transcript snippet (text of what is being taught).

    # Your task:
    # 1. Answer the user's question **only if it relates to this course**.
    # 2. If unrelated, say: "I can only answer questions about this Java course."
    # 3. If relevant:
    #     - Identify **which video(s)** contain the answer,
    #     - Give **exact timestamps** where the concept is taught,
    #     - Provide a **clear explanation** with references.

    # Here are the top matching video chunks:
    #     {json.dumps(context_json, indent=4)}

    # ------------
    # User Question: {user_query}
    # '''
#     prompt = f"""You are an AI teaching assistant for a **Java programming course**. 
# The course content is divided into video chunks, each with:
# - A title that includes the video number and name,
# - A timestamp (start and end in seconds),
# - A transcript snippet (text of what is being taught).

# Instructions:
# 1. Answer the user's question **clearly and concisely** using the provided chunks.
# 2. After the answer, provide a list of relevant **videos with exact timestamps** where the answer can be found.
# 3. If the question is **unrelated** to this course, respond: "I can only answer questions about this Java course."
# 4. Do **not** speculate or provide extra reasoning. Only use the given chunks.

# Here are the top matching video chunks (JSON format):
# {json.dumps(context_json, indent=4)}

# ------------
# User Question: {user_query}
# """
    prompt = f"""
You are an AI assistant for a Java programming course.
Answer the user's question **only** using the provided transcript chunks.

STRICT FORMAT for response:
{{
    "answer": "short direct answer here",
    "references": [
        {{
            "title": "video title",
            "start_time": "start timestamp in seconds",
            "end_time": "end timestamp in seconds",
            "snippet": "exact sentence from transcript"
        }}
    ]
}}

Rules:
- If you don't find relevant information in the chunks, respond with:
  {{"answer": "I'm sorry, I couldn't find relevant information about that in the course."}}
- Be concise.
- Use exact timestamps from the chunks.
- Maximum 3 references.

Chunks:
{json.dumps(context_json, indent=4)}

User Question: {user_query}
"""



    # Get AI response
    answer = inference(prompt, model="deepseek-r1")
    
    # Display
    st.markdown("### 📝 Answer")
    st.write(answer)
    
    st.markdown("### 🎬 Top Relevant Video Chunks")
    st.dataframe(df_sorted[['title', 'text', 'start', 'end', 'similarity']])
