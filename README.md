# Chatbot

**Chatbot** is an AI-powered chatbot built using open-source models, Python, and Streamlit. The chatbot supports interaction via a web-based UI and a CLI (Command Line Interface), and can extract content, summarize articles, and perform various tasks based on a given query.

## Table of Contents
- [Setup Instructions](#setup-instructions)
  - [Create and Activate Conda Environment](#1-create-and-activate-conda-environment)
  - [Install Dependencies](#2-install-dependencies)
  - [Download Models](#3-download-models)
  - [Running the Chatbot](#4-running-the-chatbot)
- [Folder Structure](#folder-structure)



---

## Setup Instructions

### 1. Create a Conda Environment
First, create and activate the Conda environment for the project.

### 2. Install
pip install -r requirements.txt

### 3. Model links
LLama - https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q8_0.gguf
Mistral - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q8_0.gguf
pip install -r requirements.txt

```bash
conda create --name chatbot_env python=3.9
conda activate chatbot_env
```

###Folder Structure
Chatbot-Relinns/
│
├── modules/
│   ├── models/                # Store model files here
│   │   ├── mistral-7b-v0.1.Q4_K_M.gguf  # Mistral model file
│   │   └── llama-2-7b.Q8_0.gguf         # Llama model file
│   ├── chatbot.py             # Main chatbot script
│   ├── requirements.txt       # List of Python dependencies
│   └── .env                   # Environment variables (e.g., API keys)
│
├── .gitignore                 # Git ignore file to exclude unnecessary files
└── README.md                  # Project documentation
