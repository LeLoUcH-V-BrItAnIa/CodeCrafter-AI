import streamlit as st
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="Online Compiler", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
        /* Background */
        body {
            font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
        }
        
        /* Title */
        .title {
            font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
            text-align: center;
            font-size: 3rem;
            # color: #11567f;
            font-weight: bold;
        }

        /* Code Input Box */
        .stTextArea>div>textarea {
            border-radius: 10px;
            border: 1px solid #ccc;
            padding: 15px;
            font-size: 16px;
            background-color: #ffffff;
        }

        /* Buttons */
        .stButton>button {
            font-weight: bold;
            background-color: red;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 18px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            color: black;
            background: white;
        }

        /* Selectbox / Checkbox */
        .stSelectbox select, .stCheckbox>div>label {
            font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
            font-size: 16px;
            font-weight: 500;
        }

        /* Spacer between elements */
        .stMarkdown, .stButton, .stTextArea {
            margin-bottom: 20px;
        }

    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<h1 class="title">💻SmartCompile</h1>', unsafe_allow_html=True)

# --- LANGUAGE SELECTION ---
language = st.selectbox(
    "Choose Programming Language",
    ["python", "javascript", "c", "cpp", "java"],
    key="language_select"
)

# --- CODE INPUT ---
default_code = {
    "python": "print('Hello, World!')",
    "javascript": "console.log('Hello, World!');",
    "c": "#include<stdio.h>\nint main() {\n    printf(\"Hello, World!\");\n    return 0;\n}",
    "cpp": "#include<iostream>\nusing namespace std;\nint main() {\n    cout << \"Hello, World!\";\n    return 0;\n}",
    "java": "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}"
}

# Pre-fill the text area with sample code
code_input = st.text_area("Enter your code below:", default_code[language], height=250)

# --- EXPLANATION TOGGLE ---
explain = st.checkbox("Get Explanation")

# --- COMPILE BUTTON ---
if st.button("Run Code"):
    if not code_input.strip():
        st.warning("⚠️ Please enter some code.")
    else:
        with st.spinner("Compiling with Gemini..."):
            try:
                res = requests.post(
                    "http://localhost:5000/api/run",  # Update this URL as needed
                    json={
                        "language": language,
                        "code": code_input,
                        "explain": explain
                    }
                )
                if res.ok:
                    st.success("✅ Output:")
                    st.code(res.json()["response"], language=language)
                else:
                    st.error("❌ Error: " + res.text)
            except Exception as e:
                st.error(f"❌ Request failed: {e}")