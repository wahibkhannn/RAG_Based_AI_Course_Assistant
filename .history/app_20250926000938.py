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

def expand_with_neighbors(df_sorted, window_size=1):
    """
    For each matched chunk index in df_sorted, also include
    neighboring chunks based on window_size.
    """
    # Get original indices in df
    indices = df_sorted.index.tolist()
    expanded_indices = set()

    for idx in indices:
        for i in range(idx - window_size, idx + window_size + 1):
            if 0 <= i < len(df):
                expanded_indices.add(i)

    expanded_df = df.loc[sorted(expanded_indices)]
    return expanded_df


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
    # Truncate long text fields for display
    df_sorted['text'] = df_sorted['text'].apply(lambda x: x[:700] + "..." if len(x) > 700 else x)

    # Build context
    context_json = df_sorted[['title', 'text', 'start', 'end', 'similarity']].to_dict(orient='records')
    

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


    # Get AI response
    result = inference(prompt)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = {
        "timestamp": timestamp,
        "user_query": user_query,
        "prompt": prompt,
        "raw_response": result
    }

    log_file = f"logs/query_response_{timestamp}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Saved query and response to {log_file}")


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