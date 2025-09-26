
import streamlit as st
from dotenv import load_dotenv
from src.backend import extract_video_id, fetch_transcript, build_vectorstore_from_text, build_chain, save_vectorstore, load_vectorstore_if_exists
import os
load_dotenv()

def main():
    st.set_page_config(page_title="YouTube AI Chat", page_icon="🎬", layout="centered")

    # --- NEON DARK MODE UI ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg-color: #111827;
        --container-bg: #1f2937;
        --primary-accent: #3b82f6;
        --text-color: #f9fafb;
        --subtle-text: #9ca3af;
        --border-color: #374151;
    }
    .stApp { background-color: var(--bg-color); color: var(--text-color); font-family: 'Inter', sans-serif; }
    .stApp > header { background-color: transparent; }
    .main > div { background-color: transparent; }
    .block-container { background-color: transparent; padding: 1rem !important; }
    .stTextInput > label, .stCheckbox > label { color: var(--subtle-text) !important; font-weight: 500; }
    .stTextInput > div > div > input, .stChatInput > div > div > input {
        background-color: var(--bg-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0.5rem !important;
        color: var(--text-color) !important;
    }
    .stTextInput > div > div > input:focus, .stChatInput > div > div > input:focus {
        border-color: var(--primary-accent) !important;
        box-shadow: 0 0 0 2px var(--primary-accent) !important;
    }
    .stButton > button { 
        background-color: var(--primary-accent) !important; 
        color: white !important; 
        border-radius: 0.5rem !important; 
        border: none !important; 
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover { background-color: #60a5fa !important; }
    div[data-testid="stChatMessage"] { background-color: var(--container-bg); border-radius: 0.75rem; border: 1px solid var(--border-color); }
    .stSpinner > div { color: var(--primary-accent); }
    .custom-error { background-color: #ef44441a; border: 1px solid #ef444480; color: #fca5a5; padding: 1rem; border-radius: 0.75rem; }
    .custom-success { background-color: #22c55e1a; border: 1px solid #22c55e80; color: #86efac; padding: 1rem; border-radius: 0.75rem; }
    .custom-info { background-color: #3b82f61a; border: 1px solid #3b82f680; color: #93c5fd; padding: 1rem; border-radius: 0.75rem; }
    </style>
    """, unsafe_allow_html=True)

    # Additional small overrides to guarantee visibility/contrast
    st.markdown("""
    <style>
    .stTextInput input::placeholder, .stChatInput input::placeholder { color: #9ca3af !important; opacity: 1 !important; }
    div[data-testid="stChatMessage"] { padding: 0.75rem 1rem; margin-bottom: 0.75rem; }
    a { color: #93c5fd; }
    </style>
    """, unsafe_allow_html=True)

    # --- STATE MANAGEMENT ---
    for key in ["history", "video_id", "vector_store", "loaded_url"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key == 'history' else None

    # --- API KEY CHECK ---
    google_api_key = os.getenv('GOOGLE_API')
    if not google_api_key:
        st.markdown('<div class="custom-error"><strong>ERROR:</strong> GOOGLE_API key not found. Please set it in your .env file.</div>', unsafe_allow_html=True)
        st.stop()

    # --- HEADER --- 
    st.title("🎬 YouTube AI Chat")
    st.markdown("Chat with any YouTube video instantly. Just paste the URL, load the video, and start asking questions.")

    # --- VIDEO INPUT --- 
    url_or_id = st.text_input("Enter YouTube URL or Video ID", value=st.session_state.loaded_url or "", key="url_input")
    
    if st.button("Load Video", use_container_width=True):
        if not url_or_id:
            st.markdown('<div class="custom-error">Please enter a YouTube URL or Video ID.</div>', unsafe_allow_html=True)
        else:
            video_id = extract_video_id(url_or_id)
            if not video_id:
                st.markdown('<div class="custom-error">Invalid YouTube URL or Video ID. Please check and try again.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Fetching transcript and building index..."):
                    try:
                        vector_store = load_vectorstore_if_exists(google_api_key, video_id)
                        if not vector_store:
                            transcript_text = fetch_transcript(video_id)
                            vector_store = build_vectorstore_from_text(transcript_text, google_api_key)
                            save_vectorstore(vector_store, video_id)
                        
                        st.session_state.video_id = video_id
                        st.session_state.vector_store = vector_store
                        st.session_state.loaded_url = url_or_id
                        st.session_state.history = []
                        st.markdown('<div class="custom-success">Video loaded successfully! You can now ask questions below.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="custom-error"><strong>Error loading video:</strong> {e}</div>', unsafe_allow_html=True)

    # --- CHAT INTERFACE ---
    if st.session_state.vector_store:
        st.markdown("--- ")
        # Display chat history
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("Ask a question about the video..."):
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chain = build_chain(st.session_state.vector_store, google_api_key)
                    response = chain.invoke(prompt)
                    st.markdown(response)
            
            st.session_state.history.append({"role": "assistant", "content": response})
    else:
        st.markdown('<div class="custom-info">Load a video to begin the chat.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
