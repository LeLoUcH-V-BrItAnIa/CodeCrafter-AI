import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Use your model
model_name = "gemini-2.5-flash-preview-04-17"

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

    # Initialize Gemini client
    client = genai.Client(
        api_key="AIzaSyB4BlvNvIBd8yfiVwrTrB832zixrUQAKKU",  # Replace with os.environ.get(...) for security in real apps
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
    app.run(debug=True)