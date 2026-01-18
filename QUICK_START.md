# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Set Up Environment (2 min)

```bash
# Copy environment template
cp env.example .env

# Edit with your API keys
nano .env
```

**Minimum required:**
```env
OPENAI_API_KEY=sk-...
MURF_API_KEY=...
SONIOX_API_KEY=...
```

### Step 2: Add Knowledge Base (1 min)

Your agricultural knowledge should be in `resource_document.txt`. This file is already present with agricultural product information.

### Step 3: Run the Bot (2 min)

```bash
# First run (generates embeddings, takes ~30 seconds)
uv run bot.py

# You'll see:
# 📚 Initializing RAG system with 58000 characters...
# 📦 Created 21 chunks
# 🆕 First run, generating and caching embeddings...
# ✅ RAG system ready - embeddings saved to cache!
```

### Step 4: Test It

Connect via browser at `http://localhost:7860` and ask questions in Telugu!

---

## �� What Just Happened?

1. ✅ Knowledge base split into 21 chunks
2. ✅ Embeddings generated and cached (one-time cost: ₹0.18)
3. ✅ Bot ready to answer questions
4. ✅ Future runs will load embeddings from cache (no cost)

---

## 🎯 Common Configurations

### Use Cheaper Embedding Model

Edit `.env`:
```env
EMBED_MODEL=text-embedding-3-small  # 85% cheaper
```

Delete cache and restart:
```bash
rm -rf .rag_cache
uv run bot.py
```

### Change LLM Model

```env
# Cheaper option
OPENAI_MODEL=gpt-4o-mini

# Or use Azure
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### Adjust RAG Behavior

```env
CHUNK_SIZE=500    # Larger chunks = fewer, more context per chunk
TOP_K=5           # Retrieve more context per question
```

---

## 🔍 Verify It's Working

### Check Logs

**First run should show:**
```
🆕 First run, generating and caching embeddings...
✅ RAG system ready - embeddings saved to cache!
```

**Second run should show:**
```
📂 Loading cached embeddings from .rag_cache/embeddings_abc123.npy
✅ RAG system ready - embeddings loaded from cache!
```

### Check Cache Directory

```bash
ls -lh .rag_cache/
# Should show:
# embeddings_<hash>.npy  (~1-2 MB)
# current_hash.txt       (32 bytes)
```

### Monitor Costs

**First conversation:**
- Embedding: ₹0.18 (one-time)
- Conversation: ₹0.48
- **Total: ₹0.66**

**Subsequent conversations:**
- Embedding: ₹0 (cached)
- Conversation: ₹0.48
- **Total: ₹0.48**

---

## ⚠️ Troubleshooting

### "Port already in use"

```bash
lsof -i :7860
kill -9 <PID>
```

### "OPENAI_API_KEY not set"

Add to `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
```

### Embeddings regenerate every time

Check `.rag_cache/` exists and has write permissions:
```bash
ls -la .rag_cache/
chmod 755 .rag_cache
```

### High costs

1. Verify caching works (see "Check Logs" above)
2. Use smaller model: `EMBED_MODEL=text-embedding-3-small`
3. Reduce context: `TOP_K=2`

---

## 📚 Next Steps

1. ✅ Bot is running
2. 📖 Read [README.md](README.md) for detailed info
3. 🔧 Check [CODE_STRUCTURE.md](CODE_STRUCTURE.md) to understand the code
4. ⚙️ See `env.example.updated` for all configuration options
5. 📊 Monitor costs and adjust settings as needed

---

## 💡 Pro Tips

- **First run takes longer:** Generating embeddings for the first time
- **Cache is your friend:** Never delete `.rag_cache/` unless updating knowledge base
- **Start conservative:** Use default settings, optimize based on actual costs
- **Monitor quality:** Ensure answers are relevant before optimizing for cost
- **Version control:** Never commit `.env` or `.rag_cache/` to git

---

## 📞 Need Help?

1. Check error message in terminal
2. Look in [README.md](README.md) Troubleshooting section
3. Verify all API keys are correct in `.env`
4. Check provider status pages (OpenAI, Murf, Soniox)
