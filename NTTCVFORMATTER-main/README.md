# CV Transformer — NTT DATA Format

Convert any CV/Resume into the NTT template format using AI-powered field mapping and content extraction.

## Quick Start (First Time Only)

### 1. Install Dependencies
```powershell
cd path\to\NTTCVFORMATTER-main\NTTCVFORMATTER-main
pip install -r requirements.txt
```

### 2. Download the Embedding Model Locally ⭐ IMPORTANT
This is a **one-time setup** that eliminates network timeouts:

```powershell
python download_model.py
```

**What it does:**
- Downloads ~100MB embedding model from HuggingFace (takes 1-2 minutes)
- Saves it locally in `models/all-MiniLM-L6-v2/`
- After this, the app runs **completely offline** with instant model loading

### 3. Run the App
```powershell
python -m streamlit run app.py
```

This opens the app at: **http://localhost:8501**

---

## How It Works

1. **Upload Files:**
   - NTT template (.docx)
   - Your CV (.pdf or .docx)

2. **AI Processing (3 steps):**
   - ✅ Parse template structure
   - ✅ Extract CV text
   - ✅ Match fields using HuggingFace embeddings
   - ✅ Extract content with Groq LLaMA-3.3 API

3. **Download Result:**
   - Get a .docx file in NTT format with:
     - Your profile information
     - Professional experience
     - Skills & competencies
     - Education & certifications

---

## Requirements

- Python 3.11+ 
- Internet connection (first run only for model download)
- Groq API key (free at https://console.groq.com/keys)
- 500MB disk space for the embedding model

---

## Troubleshooting

### "Cannot send a request, as the client has been closed"
**Solution:** Run `python download_model.py` to cache the model locally

### App is slow on first run
**Normal!** First run loads HuggingFace embeddings. Use local model (see step 2 above).

### Groq API errors
1. Verify API key at https://console.groq.com/keys
2. Check you have free tier quota (12k TPM limit)
3. Wait 60 seconds between retries (rate limit resets per minute)

### Model download fails
```powershell
# Clear cache and retry
rmdir $env:USERPROFILE\.cache\sentence-transformers -r
python download_model.py
```

---

## Project Structure

```
NTTCVFORMATTER-main/
├── app.py                    # Main Streamlit app
├── download_model.py         # One-time model download script
├── requirements.txt          # Python dependencies
├── models/                   # Local embedding model (generated after download_model.py)
│   └── all-MiniLM-L6-v2/    # HuggingFace model cached locally
└── README.md                # This file
```

---

## Usage Tips

- **Small CVs are faster** — Large CVs (50+ pages) may hit API limits
- **Cache exists after first setup** — Subsequent runs are instant
- **Works offline** — After running `download_model.py`, no internet needed for local processing
- **API calls use 2-pass approach** — Reduces token usage to stay within free tier

---

## Developed by
Shubham Shandilya

**Powered by:**
- HuggingFace Transformers (embeddings)
- Groq LLaMA-3.3-70b-versatile (content extraction)
- Streamlit (UI)
- python-docx (document generation)

