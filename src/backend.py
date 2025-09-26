from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import re
import tempfile
import hashlib
import streamlit as st


load_dotenv()


def extract_video_id(url_or_id: str) -> Optional[str]:
    if not url_or_id:
        return None
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url_or_id):
        return url_or_id
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
    api = YouTubeTranscriptApi()
    preferred_langs = ['en', 'en-US', 'en-GB', 'hi', 'es', 'fr', 'de']

    # 1) Fast path: try direct fetch with preferred languages
    try:
        fetched = api.fetch(video_id, languages=preferred_langs)
        data = fetched.to_raw_data()
        text = " ".join(d.get("text", "") for d in data if d.get("text"))
        if text.strip():
            return text
    except Exception:
        pass

    # 2) Use transcript list: prefer manual, then generated, then any
    try:
        tlist = api.list(video_id)
        # Manual first
        try:
            t = tlist.find_manually_created_transcript(preferred_langs)
            fetched = t.fetch()
            data = fetched.to_raw_data()
            text = " ".join(d.get("text", "") for d in data if d.get("text"))
            if text.strip():
                return text
        except Exception:
            pass
        # Generated next
        try:
            t = tlist.find_generated_transcript(preferred_langs)
            fetched = t.fetch()
            data = fetched.to_raw_data()
            text = " ".join(d.get("text", "") for d in data if d.get("text"))
            if text.strip():
                return text
        except Exception:
            pass
        # Finally iterate all
        for t in tlist:
            try:
                fetched = t.fetch()
                data = fetched.to_raw_data()
                text = " ".join(d.get("text", "") for d in data if d.get("text"))
                if text.strip():
                    return text
            except Exception:
                continue
    except Exception:
        pass

    raise RuntimeError(f"No transcript available for video {video_id}.")


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
        google_api_key=google_api_key, model="gemini-pro", temperature=0.2
    )
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain


def get_index_cache_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "yt_chatbot_indexes")


def index_path_for_video(video_id: str) -> str:
    safe_name = hashlib.sha256(video_id.encode()).hexdigest()[:16]
    return os.path.join(get_index_cache_dir(), f"faiss_{safe_name}")


def save_vectorstore(vector_store: FAISS, video_id: str):
    path = index_path_for_video(video_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vector_store.save_local(path)


def load_vectorstore_if_exists(google_api_key: str, video_id: str) -> Optional[FAISS]:
    path = index_path_for_video(video_id)
    if os.path.exists(path):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
            return FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None
