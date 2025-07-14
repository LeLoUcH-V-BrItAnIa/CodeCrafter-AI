import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Use your model
model_name = "gemma-3n-e4b-it"

@app.route('/api/run', methods=['POST'])
def run_code():
    data = request.get_json()

    language = data.get('language')
    code = data.get('code')
    explain = data.get('explain', False)

    if not language or not code:
        return jsonify({"error": "Language and code are required."}), 400

    # Construct the dynamic prompt
    prompt = f"You are an expert {language} programmer.\n\n"
    prompt += f"Here is some code:\n\n{code}\n\n"
    prompt += "Please execute this code and return the output.\n"

    if explain:
        prompt += "Also explain the output in simple terms."
    api_key = os.getenv("GOOGLE_API_KEY")  
    # Initialize Gemini client
    client = genai.Client(
        api_key="AIzaSyAYUq-ZAOpOGRjbDZkL4-_o64gE62ZRq0w",  # Replace with os.environ.get(...) for security in real apps
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    response_text = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_text += chunk.text
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": response_text})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)