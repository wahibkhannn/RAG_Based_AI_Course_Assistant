def inference(prompt):
    # The API endpoint for the Gemini Pro model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"

    # The headers, including your API key
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': GEMINI_API_KEY
    }

    # The JSON payload in the format required by the Gemini API
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        # Make the POST request to the Gemini API
        r = requests.post(url, headers=headers, json=data)
        r.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Parse the JSON response
        response_data = r.json()

        # Extract the generated text from the response
        # The response structure is nested, so you need to navigate it correctly
        output_text = response_data['candidates'][0]['content']['parts'][0]['text']

        # Save the raw JSON for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/raw_response_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(response_data, f, ensure_ascii=False, indent=4)

        return output_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini API response: {e}")
        # Log the problematic response for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/error_response_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(r.json(), f, ensure_ascii=False, indent=4)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None