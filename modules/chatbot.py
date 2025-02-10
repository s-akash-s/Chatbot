import streamlit as st
import requests
from bs4 import BeautifulSoup
from llama_cpp import Llama
import torch
import base64
import re
import openai
import argparse
import sys

# Available options
MODEL_OPTIONS = ["OpenAI", "Mistral-7B", "Llama-2-7B"]

# Open-source model paths
MODELS = {
    "Mistral-7B": "models/mistral-7b-v0.1.Q4_K_M.gguf",
    "Llama-2-7B": "models/llama-2-7b.Q8_0.gguf"
}


# Detect GPU availability
use_gpu = torch.cuda.is_available()
gpu_layers = 35 if use_gpu else 0  

# Base64 Encoding Functions
def encode_text(text):
    return base64.b64encode(text.encode()).decode()

def decode_text(encoded_text):
    return base64.b64decode(encoded_text).decode()

# Web Scraping Function
def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")  
        text_content = "\n".join(p.get_text() for p in paragraphs)
        
        text_content = re.sub(r"\s+", " ", text_content).strip()[:5000]  
        return encode_text(text_content)
    except Exception as e:
        return f"Error: {str(e)}"

# Chat Function
def chat_with_model(selected_model, user_input, chat_history, openai_api_key=None, webpage_content=""):
    system_prompt = "You are a helpful AI assistant. Keep responses concise and relevant."

    if webpage_content:
        system_prompt += f"\n\nYou have access to the following webpage content:\n\n{webpage_content[:1500]}\n\n"

    prompt = f"{system_prompt}\n\n{chat_history}\nUser: {user_input}\n\nAssistant:"

    if selected_model == "OpenAI":
        if not openai_api_key:
            return "⚠️ Error: OpenAI API key is required for OpenAI models."

        openai_client = openai.OpenAI(api_key=openai_api_key)  # ✅ Create OpenAI client properly

        try:
            response = openai_client.chat.completions.create(  # ✅ Fixed new API format
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                temperature=0.7
            ).choices[0].message.content.strip()
        except Exception as e:
            response = f"Error: {str(e)}"
    else:
        llm = Llama(
            model_path=MODELS[selected_model],
            n_gpu_layers=gpu_layers,
            n_ctx=4096,
            verbose=False
        )
        response = llm(
            prompt,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "Assistant:"]
        )["choices"][0]["text"].strip()

    return response

# Streamlit App 
def run_streamlit():
    st.sidebar.title("Settings")
    selected_model = st.sidebar.selectbox("Choose a Model", MODEL_OPTIONS)

    if selected_model == "OpenAI":
        openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    else:
        openai_api_key = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("🌐 Web Scraper & AI Chatbot")
    url_input = st.text_input("Enter a URL (optional):", "")

    if url_input.strip():
        st.session_state.encoded_content = scrape_website(url_input)
        st.success("Webpage content loaded! Ask me questions about it.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
        webpage_content = decode_text(st.session_state.encoded_content) if "encoded_content" in st.session_state else ""

        response = chat_with_model(selected_model, user_input, chat_history, openai_api_key, webpage_content)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# CLI Mode 
def run_cli():
    parser = argparse.ArgumentParser(description="CLI for Web Scraper & AI Chatbot")
    parser.add_argument("--model", type=str, choices=MODEL_OPTIONS, required=True, help="Select AI model")
    parser.add_argument("--query", type=str, help="Initial query for the chatbot")  # ✅ Add query argument
    parser.add_argument("--url", type=str, help="URL to scrape for content")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (required for OpenAI model)")

    args = parser.parse_args()

    chat_history = ""  # Stores conversation history
    webpage_content = ""

    # Scrape website if URL is provided
    if args.url:
        webpage_content = decode_text(scrape_website(args.url))
        print("\n🔗 Scraped Content (Truncated):\n", webpage_content[:1500], "\n")

    print("\n💬 Assistant: Hello! How can I help you? (Type 'exit' to quit)\n")

    # Handle initial query if provided
    if args.query:
        user_input = args.query
        print(f"You: {user_input}")
        response = chat_with_model(args.model, user_input, chat_history, args.api_key, webpage_content)
        chat_history += f"User: {user_input}\n\nAssistant: {response}\n\n"
        print(f"\n💬 Assistant: {response}\n")

    # Continue interactive chat mode
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("\n👋 Exiting chat. Have a great day!")
            break

        chat_history += f"User: {user_input}\n\n"
        response = chat_with_model(args.model, user_input, chat_history, args.api_key, webpage_content)
        chat_history += f"Assistant: {response}\n\n"

        print(f"\n💬 Assistant: {response}\n")


# Run as Streamlit App or CLI
if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        run_streamlit()
