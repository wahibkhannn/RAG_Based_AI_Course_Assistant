# app.py
import streamlit as st
import pandas as pd
import json
from process_incoming_query import create_embedding, inference, df
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import re
from dotenv import load_dotenv
load_dotenv()

def clean_json_output(raw_output):
    """
    Extracts the first JSON object found in the LLM response
    and cleans common formatting issues like trailing commas.
    """
    # Extract JSON using regex
    match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)

    # Remove trailing commas
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


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
    # df_sorted = df.sort_values(by='similarity', ascending=False).head(7)

    # Sort by similarity and take top matches
    top_matches = df.sort_values(by='similarity', ascending=False).head(7)

    # Expand with neighboring chunks
    df_sorted = expand_with_neighbors(top_matches, window_size=1)

    # Optional: truncate long text for UI
    df_sorted['text'] = df_sorted['text'].apply(lambda x: x[:700] + "..." if len(x) > 700 else x)


    df_sorted['text'] = df_sorted['text'].apply(lambda x: x[:700] + "..." if len(x) > 700 else x)

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


    prompt = f"""You are an AI assistant for an educational course.

Your task:
- Answer the user's question using the provided transcript chunks.
- **Synthesize information** from multiple chunks if needed.
- **Make reasonable inferences** from the available context.
- If you find related but not exact information, mention what you found.
- Only say no information is available if truly nothing relevant exists.

VERY IMPORTANT:
- Respond ONLY with a valid JSON object.

JSON Format:
{{
  "answer": "Your answer here, combining information from chunks as needed",
  "references": [
    {{
      "title": "video title",
      "start_time": 65.0,
      "end_time": 69.0,
      "snippet": "relevant text from transcript"
    }}
  ]
}}

Here are the most relevant transcript chunks:
{json.dumps(context_json, indent=2)}

User Question: {user_query}"""

    # prompt = f"""
    # You are an AI assistant for a Java programming course.

    # Your job is to answer the user's question using ONLY the provided transcript chunks.

    # VERY IMPORTANT:
    # - Respond ONLY with a valid JSON object.
    # - No extra text, no explanations outside of JSON.
    # - If the answer is not found in the chunks, return:
    #   {{
    #     "answer": "I'm sorry, I couldn't find relevant information about that in the course.",
    #     "references": []
    #   }}

    # JSON Format:
    # {{
    #   "answer": "short direct answer here",
    #   "references": [
    #     {{
    #       "title": "video title",
    #       "start_time": 65.0,
    #       "end_time": 69.0,
    #       "snippet": "exact sentence from transcript"
    #     }}
    #   ]
    # }}

    # Here are the transcript chunks:
    # {json.dumps(context_json, indent=4)}

    # User Question: {user_query}
    # """




    # Get AI response
    # Get AI response
    result = inference(prompt)

    if not result:
        st.error("Failed to get response from Gemini API. Check your API key or permissions.")
    else:
        # Clean and parse JSON
        data = clean_json_output(result)

        if data is None:
            st.error("Invalid response format. Check the prompt or model output.")
        else:
            st.markdown("### 📝 Answer")
            st.write(data['answer'])

            if data['references']:
                st.subheader("References")
                for ref in data['references']:
                    st.write(f"**{ref['title']}** ({ref['start_time']}s - {ref['end_time']}s)")
                    st.write(f"> {ref['snippet']}")
            else:
                st.write("No references found.")