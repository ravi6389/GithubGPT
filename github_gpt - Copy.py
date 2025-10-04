import streamlit as st
import requests
import base64
import os
from typing import List, Tuple
import numpy as np
import faiss
import torch
# LangChain imports
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

import os

from langchain.document_loaders import TextLoader

from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from langchain.embeddings import HuggingFaceEmbeddings

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain




from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from pypdf import PdfReader

from langchain.chains.retrieval import create_retrieval_chain





from sentence_transformers import SentenceTransformer, util
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama

import transformers
import torch
from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForCausalLM


#from hugchat import hugchat
from typing import Any, Dict, List
import streamlit as st


# ----------------------------
# Environment & API Keys
# ----------------------------


GITHUB_TOKEN = st.secrets['GITHUB_TOKEN']
GITHUB_API = "https://api.github.com"

GROQ_API_KEY = st.secrets['GROQ_API_KEY']
llm = ChatGroq(temperature=0.8, groq_api_key=GROQ_API_KEY,
               model_name="llama-3.1-8b-instant", streaming=True)

max_repos = 20
max_files = 100

# ----------------------------
# HuggingFace Embeddings
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device}
)

# ----------------------------
# GitHub Helper Functions
# ----------------------------
def github_get(url):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response

def fetch_user_and_repos(username: str, max_repos: int = 5):
    user_url = f"{GITHUB_API}/users/{username}"
    repos_url = f"{GITHUB_API}/users/{username}/repos?per_page={max_repos}"
    user = github_get(user_url).json()
    repos = github_get(repos_url).json()
    return user, repos

def fetch_repo_files(owner: str, repo: str, path: str = ""):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    return github_get(url).json()

def fetch_file_content(owner: str, repo: str, path: str) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    data = github_get(url).json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
    return data.get("content", "")

# ----------------------------
# Corpus builder
# ----------------------------
def build_corpus_from_repos(username, repos, max_files=50):
    docs = []
    allowed_exts = (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rb", ".php", ".md")
    for repo in repos:
        try:
            files = fetch_repo_files(username, repo["name"])
            count = 0
            for f in files:
                if f["type"] == "file" and f["name"].endswith(allowed_exts):
                    text = fetch_file_content(username, repo["name"], f["path"])
                    if text:
                        docs.append((f"{repo['name']}/{f['path']}", text))
                        count += 1
                if count >= max_files:
                    break
        except Exception as e:
            print(f"Error fetching repo {repo['name']}: {e}")
    return docs

# ----------------------------
# FAISS Vector Index
# ----------------------------
def build_faiss_index(docs: List[Tuple[str, str]]):
    texts = [text for _, text in docs]
    embeddings = embeddings_model.embed_documents(texts)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_top_docs(query: str, docs, index, k=3):
    query_vec = np.array(embeddings_model.embed_query(query)).astype("float32")
    distances, indices = index.search(query_vec.reshape(1, -1), k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        doc_name, doc_text = docs[idx]
        results.append((doc_name, doc_text, 1/(1+dist)))  # similarity score
    return results

# ----------------------------
# LLM Prompt
# ----------------------------
prompt_template = PromptTemplate(
    input_variables=["user_info", "contexts", "question"],
    template="""You are a helpful assistant. Use ONLY the information in the contexts and user info to answer.

If you do not know the answer from the provided information, respond: "I don't know".

You may use both:
- User info (GitHub profile metadata, number of repos, followers, etc.)
- Repo/file contexts (code + README text)

Rules:
- If the user asks to **see the code** or the **contents of a file**, you MUST include the actual code snippet from the context. 
- Otherwise, just summarize or explain in natural language.
- Always cite which repo and file the information came from.

User Info:
{user_info}

Repo Contexts:
{contexts}

Question:
{question}

Answer:"""
)

def answer_with_llm(question: str, user_info: str, contexts: List[Tuple[str, str, float]]) -> str:
    context_block = "\n\n".join([
        f"---\nRepo/File: {name}\nRelevance: {score:.3f}\n\n{text[:2000]}"
        for name, text, score in contexts
    ])
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    return llm_chain.run({
        "user_info": user_info,
        "contexts": context_block,
        "question": question
    })

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="GitHub LLM Chatbot", layout="wide")
st.title("🤖 GitHub LLM Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I can answer questions about GitHub users, their repos, and their code. Enter a username to begin."}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

username = st.text_input("GitHub username to crawl", value="octocat")

# ----------------------------
# Crawl and index
# ----------------------------
if st.button("Crawl profile and build index") and username:
    try:
        with st.spinner("Fetching GitHub profile and repos..."):
            user, repos = fetch_user_and_repos(username, max_repos)
            st.session_state["repos"] = repos
            st.session_state["user_info"] = f"Login: {user.get('login')}, Public repos: {user.get('public_repos')}, Followers: {user.get('followers')}, Following: {user.get('following')}"
            st.success(f"Found user: {user.get('login')} — {user.get('public_repos')} public repos")
        
        with st.spinner("Fetching READMEs and code, building corpus..."):
            docs = build_corpus_from_repos(username, repos, max_files)
            st.session_state['docs'] = docs
        
        with st.spinner("Embedding documents and building FAISS index..."):
            index, embeddings = build_faiss_index(docs)
            st.session_state['index'] = index
            st.session_state['embeddings'] = embeddings
            st.success(f"Indexed {len(docs)} files (README + code)")
    except Exception as e:
        st.error(f"Error: {e}")

# ----------------------------
# Repo Explorer (folders/files only, no code shown)
# ----------------------------
if "repos" in st.session_state:
    st.subheader("📂 Explore Repositories")
    repo_names = [r["name"] for r in st.session_state["repos"]]
    selected_repo = st.selectbox("Select a repository", repo_names)
    if selected_repo:
        try:
            files = fetch_repo_files(username, selected_repo)
            for f in files:
                if f["type"] == "dir":
                    st.text(f"📁 {f['path']} (folder)")
                elif f["type"] == "file":
                    st.text(f"📄 {f['path']}")
        except Exception as e:
            st.error(f"Error loading repo contents: {e}")

# ----------------------------
# Chat interaction
# ----------------------------
if 'docs' in st.session_state:
    st.subheader("💬 Chat with the LLM")
    if prompt := st.chat_input("Ask a question about this GitHub user or repos"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Retrieving relevant docs and asking LLM..."):
            top = retrieve_top_docs(prompt, st.session_state['docs'], st.session_state['index'], k=3)
            answer = answer_with_llm(prompt, st.session_state["user_info"], top)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

