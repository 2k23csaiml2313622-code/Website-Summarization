import os
import validators
import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================================================
# ENV
# ==================================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ==================================================
# UI
# ==================================================
st.set_page_config(
    page_title="Website Summarizer",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Website Summarizer")
st.caption("Summarize long articles using LLMs with chunking")

url = st.text_input("Paste a website URL")

# ==================================================
# LLM
# ==================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

chunk_prompt = PromptTemplate.from_template(
    "Summarize the following content briefly:\n\n{text}"
)

final_prompt = PromptTemplate.from_template(
    "Combine the following summaries into a clear final summary:\n\n{text}"
)

# ==================================================
# WEBSITE CONTENT LOADER
# ==================================================
def load_website_chunks(url: str):
    response = requests.get(
        url,
        timeout=15,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_text(text)

# ==================================================
# HIERARCHICAL SUMMARIZATION
# ==================================================
def hierarchical_summarize(chunks):
    partial_summaries = []

    for chunk in chunks:
        resp = llm.invoke(chunk_prompt.format(text=chunk))
        partial_summaries.append(resp.content)

    combined = "\n".join(partial_summaries)
    final = llm.invoke(final_prompt.format(text=combined))

    return final.content

# ==================================================
# ACTION
# ==================================================
if st.button("Summarize"):
    if not validators.url(url):
        st.error("Please enter a valid website URL")
    else:
        try:
            with st.spinner("Summarizing website content..."):
                chunks = load_website_chunks(url)
                summary = hierarchical_summarize(chunks)

                st.success("Summary generated successfully!")
                st.write(summary)

        except Exception as e:
            st.error(f"Error: {e}")
