# FarmVaidya AI Voice Bot (RAG-Enabled)

FarmVaidya AI Voice Bot is a **production-ready, multilingual voice AI assistant** built using **Pipecat** and enhanced with **Retrieval-Augmented Generation (RAG)**.

This system is designed specifically for **Biolmin use cases**, providing **natural spoken Telugu responses** strictly grounded in a verified knowledge base.

---

## 🌾 Overview

The bot acts like a **virtual field scientist**, interacting with farmers via voice. It answers **only from the configured knowledge base**, asks clarification questions when needed, and avoids hallucinations completely.

### Primary Focus
- ✅ Accuracy
- 🗣️ Spoken farmer-friendly Telugu
- 📚 Knowledge-based answers
- ⚡ Real-time voice interaction

---

## ✨ Key Features

- 🎤 **Real-time Voice Conversations**
  - WebRTC / Daily transport
  
- 🧠 **RAG-Based Knowledge Retrieval**
  - Answers strictly from documents
  
- 🌐 **Multilingual Architecture**
  - Telugu (default), extensible to English, Tamil, Hindi
  
- 🔄 **Embedding Cache**
  - One-time embedding generation
  - Faster restarts
  
- 🔌 **Pluggable Providers**
  - **LLM:** OpenAI, Azure OpenAI, Google Gemini  
  - **STT:** Soniox, Deepgram, Sarvam  
  - **TTS:** Murf, Sarvam, Azure, Cartesia
  
- 🗣️ **Natural Farmer Telugu**
  - No formal or bookish language
  
- 🎯 **STT Context Injection**
  - Agriculture-specific terms for better accuracy
  
- 🛑 **Hallucination Control**
  - Answers only from knowledge base
  - Polite clarification when data is missing

---

### Prerequisites

- Python 3.12+
- API Keys:
  - OpenAI (required for embeddings)
  - One STT provider
  - One TTS provider
  - Optional: Gemini / Azure OpenAI

---

## 🚀 Quick Start


### 1. Clone Repository

```bash
git clone https://github.com/farmvaidya-ai/pipecat_agent_v5.git
cd pipecat_agent_v5
```

### 2. Install Dependencies

**Using uv (Recommended - Fast):**

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

**Using pip (Alternative):**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Configure Environment Variables

```bash
# Copy example environment file
cp env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

### 4. Run the Bot

```bash
# Using uv
uv run bot.py

# Using python directly
python bot.py
```

**Expected Output:**

```
🚀 Starting Pipecat bot...
⏳ Loading models and imports (20 seconds, first run only)
✅ All components loaded successfully!
✅ Loaded knowledge base: 180000 characters
🌐 Bot running at http://localhost:7860
```

---

## ⚙️ Environment Configuration

### 🔑 Required Key

```env
OPENAI_API_KEY=your_openai_key_here
```

> **Note:** OpenAI key is mandatory for embeddings even if Gemini or Azure is used.

### 🧠 LLM Configuration

```env
LLM_PROVIDER=gemini   # Options: openai | azure | gemini
```

**Gemini:**
```env
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-1.5-pro
```

**Azure OpenAI:**
```env
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### 🎙️ Speech-to-Text (STT)

```env
STT_PROVIDER=sarvam   # Options: soniox | deepgram | sarvam
```

**Soniox:**
```env
SONIOX_API_KEY=your_soniox_key
SONIOX_MODEL=stt-rt-v3
```

**Deepgram:**
```env
DEEPGRAM_API_KEY=your_deepgram_key
```

**Sarvam:**
```env
SARVAM_API_KEY=your_sarvam_key
```

### 🔊 Text-to-Speech (TTS)

```env
TTS_PROVIDER=murf    # Options: murf | sarvam | cartesia | azure
```

**Murf (Telugu Voice):**
```env
MURF_API_KEY=your_murf_key
MURF_VOICE_ID=Ronnie   # Options: Zion/Josie/Ronnie (Telugu), Aarav (English)
MURF_STYLE=Conversational
MURF_MODEL=FALCON
MURF_REGION=in
```

**Azure TTS:**
```env
AZURE_API_KEY=your_azure_key
AZURE_REGION=eastus
AZURE_VOICE=te-IN-MohanNeural
```

**Cartesia:**
```env
CARTESIA_API_KEY=your_cartesia_key
```

### 📚 Knowledge Base

```env
KNOWLEDGE_FILE=resource_document.txt
```

- Plain text document
- Used strictly for answering queries

### 🔄 RAG Configuration

```env
EMBED_MODEL=text-embedding-3-large
CHUNK_SIZE=400
TOP_K=6
MIN_CHUNKS=5
SIMILARITY_THRESHOLD=0.3
RAG_CACHE_DIR=.rag_cache
```

- Embeddings cached automatically
- Cache regenerates only when document changes

### 🌐 Transport

```env
DAILY_API_KEY=your_daily_key
```

Supports:
- WebRTC
- Daily audio rooms

---

## 🧠 How RAG Works

1. **User speaks** → STT converts voice to text
2. **Query embedding** is generated
3. **Relevant document chunks** are retrieved
4. **Retrieved context** injected into system prompt
5. **LLM responds** strictly from context
6. **Response** converted back to speech

---

## 🗂️ Project Structure

```
pipecat_agent_v5/
├── bot.py                    # Main bot application
├── .env                      # Your API keys (not committed)
├── env.example               # Environment template
├── resource_document.txt     # Knowledge base
├── .rag_cache/              # Embeddings cache (auto-generated)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── CODE_STRUCTURE.md         # Detailed code documentation
```

---

## 🔒 Security Best Practices

- ❌ Never commit `.env`
- 🔄 Rotate API keys regularly
- ❌ Do not commit `.rag_cache/`
- 📊 Restrict API usage limits
- ✅ Keep knowledge base verified

---

## 🧪 Testing Tips

- **First run** → embeddings generated
- **Next runs** → cache loaded
- **Editing knowledge file** → cache regenerated
- Logs clearly show RAG and STT behavior

---

## 📞 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/farmvaidya-ai/pipecat_agent_v5/issues)
- Contact the FarmVaidya AI team

---

