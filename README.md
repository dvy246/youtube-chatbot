# youtube-chatbot

## Streamlit App

Run a YouTube transcript-grounded QA chatbot powered by Gemini.

### Prerequisites
- Python 3.10+
- Mistral Ai is used so get ur api key

### Setup
1. Create and activate a virtualenv (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your environment variable in a `.env` file (same directory) or via shell:
   ```bash
   echo "mistral_ai=your_api_key_here" > .env
   # or export mistral_ai=your_api_key_here
   ```

### Run
```bash
streamlit run app.py
```

Then open the URL shown by Streamlit. Enter a YouTube URL or video ID, provide your question, and ask. You can also paste your Google API key in the sidebar if not set in `.env`.

### Notes
- FAISS indexes are cached per-video under your system temp directory to speed up repeated queries.
- Supported transcript languages default to English and Hindi. Adjust in `app.py` if needed.
