import os
import re
import time
import tempfile
import hashlib
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, YouTubeRequestFailed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


def extract_video_id(url_or_id: str) -> Optional[str]:
    if not url_or_id:
        return None
    # If already looks like an ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url_or_id):
        return url_or_id
    # Try to extract from common YouTube URL formats
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id=video_id, languages=["en", "hi"]
        )
        transcript = " ".join([item["text"] for item in transcript_list])
        return transcript
    except YouTubeRequestFailed as e:
        raise RuntimeError(f"Transcript fetch failed: {e}") from e


def build_vectorstore_from_text(text: str, google_api_key: str) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([text])
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=google_api_key, model="models/embedding-001"
    )
    vector_store = FAISS.from_documents(embedding=embeddings, documents=chunks)
    return vector_store


def build_chain(vector_store: FAISS, google_api_key: str):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant.\n"
            "Answer ONLY from the provided transcript context.\n"
            "If the context is insufficient, say you don't know.\n\n"
            "{context}\n"
            "Question: {question}"
        ),
        input_variables=["context", "question"],
    )

    def format_context(response):
        return "\n\n".join([doc.page_content for doc in response])

    parallel_chain = RunnableParallel(
        {"context": retriever | RunnableLambda(format_context), "question": RunnablePassthrough()}
    )

    llm = GoogleGenerativeAI(
        google_api_key=google_api_key, model="gemini-2.5-flash", temperature=0.2
    )
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain


def get_index_cache_dir() -> str:
    base = os.path.join(tempfile.gettempdir(), "yt_chatbot_indexes")
    os.makedirs(base, exist_ok=True)
    return base


def index_path_for_video(video_id: str) -> str:
    safe = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:16]
    return os.path.join(get_index_cache_dir(), f"faiss_{safe}")


def save_vectorstore(vector_store: FAISS, video_id: str) -> None:
    path = index_path_for_video(video_id)
    vector_store.save_local(path)


def load_vectorstore_if_exists(google_api_key: str, video_id: str) -> Optional[FAISS]:
    path = index_path_for_video(video_id)
    if not os.path.exists(path):
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key, model="models/embedding-001"
        )
        return FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None


def main():
    st.set_page_config(page_title="YouTube QA Chatbot", page_icon="🎥", layout="wide")

    st.title("🎥 YouTube QA Chatbot")
    st.caption("Ask questions grounded in a video's transcript")

    with st.sidebar:
        st.subheader("Settings")
        default_key = os.getenv("GOOGLE_API", "")
        google_api_key = st.text_input(
            "Google API key (Gemini)", type="password", value=default_key, help="Env var GOOGLE_API"
        )
        persist_index = st.checkbox("Cache index for video", value=True)
        st.markdown(
            "Need an API key? See Google AI Studio.")

    url_or_id = st.text_input("YouTube URL or Video ID", placeholder="https://youtu.be/XXXXXXXXXXX or ID")
    question = st.text_input("Your question", placeholder="What is the video about?")

    if "history" not in st.session_state:
        st.session_state.history = []

    col1, col2 = st.columns([1, 1])
    with col1:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.history = []
        st.experimental_rerun()

    if ask_clicked:
        if not google_api_key:
            st.error("Please provide your Google API key.")
            st.stop()
        video_id = extract_video_id(url_or_id)
        if not video_id:
            st.error("Enter a valid YouTube URL or 11-char video ID.")
            st.stop()
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        with st.spinner("Preparing transcript and index..."):
            vector_store = None
            if persist_index:
                vector_store = load_vectorstore_if_exists(google_api_key, video_id)
            if vector_store is None:
                transcript = fetch_transcript(video_id)
                vector_store = build_vectorstore_from_text(transcript, google_api_key)
                if persist_index:
                    save_vectorstore(vector_store, video_id)

        chain = build_chain(vector_store, google_api_key)
        with st.spinner("Thinking..."):
            try:
                start = time.time()
                answer = chain.invoke(question)
                elapsed = time.time() - start
            except Exception as e:
                st.error(f"Model call failed: {e}")
                st.stop()

        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "video_id": video_id,
            "latency_sec": round(elapsed, 2),
        })

    for turn in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.caption(f"Video: {turn['video_id']} · {turn['latency_sec']}s")


if __name__ == "__main__":
    main()


