#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example with RAG.

The example runs a simple voice AI bot with Retrieval-Augmented Generation
to reduce token usage by 90% (from ~15,000 to ~1,500 tokens per conversation).
"""

import os
import re
import hashlib
import numpy as np
import PyPDF2

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat_murf_tts import MurfTTSService
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.google.llm import GoogleLLMService
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import Frame, TranscriptionFrame
import httpx

class AzureOpenAILLMService(OpenAILLMService):
    """Custom OpenAI LLM Service that supports Azure OpenAI with api-version."""
    
    def __init__(self, *, azure_api_version=None, **kwargs):
        self._azure_api_version = azure_api_version
        super().__init__(**kwargs)
    
    def create_client(self, api_key=None, base_url=None, organization=None, 
                     project=None, default_headers=None, **kwargs):
        """Override to add default_query for Azure API version."""
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "organization": organization,
            "project": project,
            "http_client": DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100, 
                    max_connections=1000, 
                    keepalive_expiry=None
                )
            ),
            "default_headers": default_headers,
        }
        
        if self._azure_api_version:
            client_kwargs["default_query"] = {"api-version": self._azure_api_version}
        
        return AsyncOpenAI(**client_kwargs)

logger.info("✅ All components loaded successfully!")

# ===== LOAD ENVIRONMENT VARIABLES =====
load_dotenv(override=True)

# ===== RAG CONFIGURATION (from environment variables) =====
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")  # Embedding model for RAG
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))  # Words per chunk
TOP_K = int(os.getenv("TOP_K", "6"))  # Number of chunks to retrieve (increased for better coverage)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))  # Minimum similarity score (0-1, lower = more permissive)
MIN_CHUNKS = int(os.getenv("MIN_CHUNKS", "5"))  # Minimum chunks to return even with low scores (fallback for STT errors)
MIN_CHUNKS = int(os.getenv("MIN_CHUNKS", "5"))  # Minimum chunks to return even with low scores (fallback for STT errors)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))  # Minimum similarity score (0-1, lower = more permissive)
RAG_CACHE_DIR = os.getenv("RAG_CACHE_DIR", ".rag_cache")  # Directory for caching embeddings

# ===== RAG UTILITY FUNCTIONS =====

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= chunk_size:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks

def embed_texts(texts, api_key):
    """Generate embeddings for a list of texts using OpenAI."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding) for d in resp.data]

def cosine_sim(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ===== HELPER FUNCTIONS =====

def extract_agricultural_terms(knowledge_base: str, max_terms: int = 50) -> list:
    """Extract important agricultural terms from knowledge base for STT context."""
    # Common agricultural terms to look for
    patterns = [
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # Capitalized words (product names)
        r'(?:cotton|tomato|chilli|paddy|rice|wheat|maize|sorghum)',  # Crops
        r'(?:fertilizer|pesticide|herbicide|fungicide|insecticide)',  # Products
        r'(?:nitrogen|phosphorus|potash|zinc|iron|calcium|magnesium)',  # Nutrients
    ]
    
    terms = set()
    for pattern in patterns:
        matches = re.findall(pattern, knowledge_base, re.IGNORECASE)
        terms.update(matches)
    
    # Predefined important terms
    important_terms = [
        "Biofactor", "NUTRI6", "Formula 6", "FLOWMIN", "Sampoorna",
        "Agriseal", "BOC", "Invictus", "Trichoderma", "Native Neem",
        "TRAICORE", "Pentazia", "NEOLIFE", "DFUSE", "DFNDR",
        "cotton", "tomato", "chilli", "paddy", "rice", "wheat", "maize",
        "fertilizer", "pesticide", "herbicide", "fungicide", "insecticide",
        "nitrogen", "phosphorus", "potash", "micronutrient", "foliar",
        "drip irrigation", "soil health", "crop yield", "bio-fertilizer"
    ]
    
    terms.update(important_terms)
    
    # Return top terms (sorted by length for better STT context)
    sorted_terms = sorted(list(terms), key=len, reverse=True)
    return sorted_terms[:max_terms]


def build_soniox_context(knowledge_base: str) -> dict:
    """Build Soniox STT context from knowledge base."""
    # Extract terms
    terms = extract_agricultural_terms(knowledge_base)
    
    # Build context object
    context = {
        "general": [
            {"key": "domain", "value": "Agriculture"},
            {"key": "topic", "value": "Agricultural products and farming consultation"},
            {"key": "language", "value": "Telugu with agricultural terminology"},
            {"key": "setting", "value": "Farmer helpline and product information"}
        ],
        "text": "This is an agricultural consultation bot providing information about bio-fertilizers, organic farming products, and crop management. The bot assists farmers with questions about Biofactor products including NUTRI6, FLOWMIN, Sampoorna, Agriseal, and other agricultural inputs.",
        "terms": terms
    }
    
    logger.info(f"🎯 Built Soniox context with {len(terms)} terms")
    return context

# ===== RAG SYSTEM CLASS =====

class SimpleRAG:
    """Simple RAG system for retrieving relevant document chunks."""
    def __init__(self, documents: str, api_key: str, cache_dir=None):
        if cache_dir is None:
            cache_dir = RAG_CACHE_DIR
        logger.info(f"📚 Initializing RAG system with {len(documents)} characters...")
        self.chunks = chunk_text(documents)
        logger.info(f"📦 Created {len(self.chunks)} chunks")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate hash of the document to detect changes
        doc_hash = hashlib.md5(documents.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"embeddings_{doc_hash}.npy")
        hash_file = os.path.join(cache_dir, "current_hash.txt")
        
        # Check if cached embeddings exist and document hasn't changed
        if os.path.exists(cache_file) and os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                cached_hash = f.read().strip()
            
            if cached_hash == doc_hash:
                logger.info(f"📂 Loading cached embeddings from {cache_file}")
                self.embeddings = np.load(cache_file, allow_pickle=True).tolist()
                logger.info(f"✅ RAG system ready - embeddings loaded from cache!")
            else:
                logger.info(f"📝 Knowledge base changed, regenerating embeddings...")
                self.embeddings = embed_texts(self.chunks, api_key)
                np.save(cache_file, self.embeddings)
                with open(hash_file, 'w') as f:
                    f.write(doc_hash)
                logger.info(f"✅ RAG system ready - embeddings saved to cache!")
        else:
            logger.info(f"🆕 First run, generating and caching embeddings...")
            self.embeddings = embed_texts(self.chunks, api_key)
            np.save(cache_file, self.embeddings)
            with open(hash_file, 'w') as f:
                f.write(doc_hash)
            logger.info(f"✅ RAG system ready - embeddings saved to cache!")
    def retrieve(self, query: str, api_key: str, k=TOP_K, threshold=SIMILARITY_THRESHOLD):
        """Retrieve top k most relevant chunks for a query with similarity threshold."""
        # Preprocess query to handle common STT errors
        processed_query = self._preprocess_query(query)
        if processed_query != query:
            logger.info(f"🔄 Query preprocessed: '{query}' → '{processed_query}'")
        
        q_emb = embed_texts([processed_query], api_key)[0]
        scored = [
            (cosine_sim(q_emb, emb), chunk)
            for emb, chunk in zip(self.embeddings, self.chunks)
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Filter by similarity threshold
        filtered = [(score, chunk) for score, chunk in scored if score >= threshold]
        
        # Robust fallback for STT errors - always return minimum chunks
        min_chunks = 5  # Always return at least 5 chunks for better coverage
        
        if filtered and len(filtered) >= min_chunks:
            # Normal case: enough chunks pass threshold
            logger.info(f"🎯 RAG Scores: Top={filtered[0][0]:.3f}, Avg={sum(s for s,_ in filtered[:k])/min(k,len(filtered)):.3f}, Retrieved={min(k, len(filtered))} chunks")
            results = filtered[:k]
        elif filtered and len(filtered) < min_chunks:
            # Some pass threshold but not enough - take top min_chunks regardless
            logger.warning(f"⚠️  Only {len(filtered)} chunks above threshold {threshold:.2f}. Using top {min_chunks} (best: {scored[0][0]:.3f})")
            results = scored[:min_chunks]
        else:
            # Nothing passes threshold - likely STT error, use permissive fallback
            logger.warning(f"🔴 No chunks above threshold {threshold:.2f}! Best score: {scored[0][0]:.3f}")
            logger.warning(f"💡 Possible STT error. Using top {min_chunks} chunks as fallback")
            results = scored[:min_chunks]
        
        # Ensure we return at least min_chunks (critical for STT errors)
        if len(results) < min_chunks:
            results = scored[:min_chunks]
            logger.info(f"📦 Fallback: Returning {len(results)} chunks to ensure minimum coverage")
        
        context = "\n\n---\n\n".join(f"[Relevance: {score:.2f}]\n{chunk}" for score, chunk in results)
        return context
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to fix common STT errors and transliteration issues."""
        # Convert to lowercase for processing
        q = query.lower()
        
        # Common Telugu-English transliteration fixes
        replacements = {
            # Common agricultural terms
            'coten': 'cotton',
            'coton': 'cotton',
            'cottan': 'cotton',
            'tometo': 'tomato',
            'tamato': 'tomato',
            'tommato': 'tomato',
            'chilly': 'chilli',
            'chili': 'chilli',
            'chillie': 'chilli',
            'fertlizer': 'fertilizer',
            'fertiliser': 'fertilizer',
            'fertilezer': 'fertilizer',
            'pesticide': 'pesticide',
            'pestisde': 'pesticide',
            'pestiside': 'pesticide',
            'herbisde': 'herbicide',
            'herbicide': 'herbicide',
            'insectisde': 'insecticide',
            'insektiside': 'insecticide',
            # Common Telugu words
            'panta': 'panta',  # crop in Telugu
            'raktalu': 'raktalu',  # types
            'vidhanam': 'vidhanam',  # method
            'fam vedja': 'farmvaidya',
            'fum vedja':'farmvaidya',
            'fum vidyaa':'farmvaidya',
            'farm vidya':'farmvaidya',
            'farm vedya':'farmvaidya',
            'farm vaidya':'farmvaidya',
            'farm media':'farmvaidya',
            'defender' : 'DFNDR'
        }
        
        # Apply replacements
        for wrong, correct in replacements.items():
            q = re.sub(r'\b' + wrong + r'\b', correct, q)
        
        # Remove extra spaces
        q = ' '.join(q.split())
        
        return q


# ===== TTS PREPROCESSING =====

def preprocess_tts_text(text: str) -> str:
    """Preprocess text before TTS to ensure correct pronunciation."""
    if not text:
        return text
    
    # Pronunciation fixes for TTS
    replacements = {
        # Acronyms and brand names - expand for better pronunciation
        'DFNDR': 'Defender',
        'DFUSE': 'D-Fuse',
        'BOC': 'B-O-C',
        'NUTRI6': 'Nutri Six',
        'FLOWMIN': 'Flow Min',
        
        # Company name variations
        'farmvaidya': 'Farm Vaidya',
        'FarmVaidya': 'Farm Vaidya',
        'FARMVAIDYA': 'Farm Vaidya',
        
        # Technical terms - phonetic spelling for clarity
        'Trichoderma': 'Tryko-derma',
        'Azospirillum': 'Azo-spirillum',
        'Pseudomonas': 'Sudo-monas',
        
        # Product names - ensure proper pronunciation
        'Biofactor': 'Bio Factor',
        'Agriseal': 'Agri Seal',
        'TRAICORE': 'Try Core',
        'Pentazia': 'Penta-zia',
        'NEOLIFE': 'Neo Life',
        'Sampoorna': 'Sampoorna',
        'Invictus': 'Invictus',
        
        # Measurements and units
        'ml': 'milliliter',
        'ML': 'milliliter',
        'kg': 'kilogram',
        'KG': 'kilogram',
        'gm': 'gram',
        'GM': 'gram',
        'ltr': 'liter',
        'LTR': 'liter',
        
        # Telugu words 
        'ఎఫ్‌వైఎమ్': 'పశువుల ఎరువు',
        'మట్టి': 'నేల',
        'శిలీంద్ర వ్యాధికారక క్రిములను':'శిలీంద్రాలను',
    }
    
    # Apply replacements (case-sensitive to preserve Telugu text)
    processed_text = text
    for original, replacement in replacements.items():
        # Use word boundary regex for whole word replacement
        processed_text = re.sub(r'\b' + re.escape(original) + r'\b', replacement, processed_text)
    
    return processed_text


# ===== CONVERSATION STATE MANAGEMENT =====

class ConversationState:
    def __init__(self):
        self.greeted = False
        self.name = None
        self.phone = None
        self.ask_count = 0
        self.terminate = False

    def has_user_info(self):
        return self.name is not None and self.phone is not None

    def try_extract_info(self, text):
        phone_match = re.search(r'\d{10,}', text)
        name_match = re.search(r'(?:my name is|i am|this is)\s+([A-Za-z ]+)', text, re.I)
        if phone_match:
            self.phone = phone_match.group(0)
        if name_match:
            self.name = name_match.group(1).strip()
        return self.has_user_info()

    def increment_ask(self):
        self.ask_count += 1
        if self.ask_count >= 2 and not self.has_user_info():
            self.terminate = True

# ===== RAG FRAME PROCESSOR =====

class RAGContextInjector(FrameProcessor):
    """Processor that injects RAG context into user messages."""
    def __init__(self, rag_system, api_key, context_messages):
        super().__init__()
        self.rag_system = rag_system
        self.api_key = api_key
        self.context_messages = context_messages
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and inject RAG context for user transcriptions."""
        await super().process_frame(frame, direction)
        
        # Check if this is a user transcription
        if isinstance(frame, TranscriptionFrame):
            text = frame.text
            if text and len(text.strip()) > 0:
                try:
                    # Retrieve relevant context
                    relevant_context = self.rag_system.retrieve(text, self.api_key, k=TOP_K, threshold=SIMILARITY_THRESHOLD)
                    logger.info(f"📚 RAG retrieved {len(relevant_context)} chars for: '{text[:50]}...'")
                    
                    # Update system message with retrieved context
                    if self.context_messages and self.context_messages[0].get("role") == "system":
                        original_content = self.context_messages[0]["content"]
                        # Reset to template first
                        if "[Context will be added per query]" not in original_content:
                            # Find the context section and replace it
                            start_marker = "Knowledge Base (relevant context provided per query):"
                            end_marker = "Rules:"
                            start_idx = original_content.find(start_marker)
                            end_idx = original_content.find(end_marker)
                            if start_idx != -1 and end_idx != -1:
                                # Replace everything between markers with placeholder
                                before = original_content[:start_idx + len(start_marker)]
                                after = original_content[end_idx:]
                                original_content = before + "\n    [Context will be added per query]\n\n    " + after
                                self.context_messages[0]["content"] = original_content
                        
                        # Now inject the relevant context
                        self.context_messages[0]["content"] = self.context_messages[0]["content"].replace(
                            "[Context will be added per query]",
                            relevant_context
                        )
                except Exception as e:
                    logger.warning(f"⚠️  RAG retrieval failed: {e}")
        
        await self.push_frame(frame, direction)


class TTSPreprocessor(FrameProcessor):
    """Processor that preprocesses text before TTS for correct pronunciation."""
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and preprocess text before TTS."""
        await super().process_frame(frame, direction)
        
        # Import necessary frame types
        from pipecat.frames.frames import TextFrame
        
        # Check if this is a text frame going to TTS
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            original_text = frame.text
            if original_text and len(original_text.strip()) > 0:
                processed_text = preprocess_tts_text(original_text)
                
                if processed_text != original_text:
                    logger.info(f"🔊 TTS preprocessed: '{original_text[:50]}...' → '{processed_text[:50]}...'")
                    frame.text = processed_text
        
        await self.push_frame(frame, direction)


# ===== BOT INITIALIZATION AND SETUP =====

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # ===== KNOWLEDGE BASE LOADING (REQUIRED FOR STT CONTEXT) =====
    knowledge_base = ""
    knowledge_file = os.getenv("KNOWLEDGE_FILE", "resource_document.txt")
    
    if os.path.exists(knowledge_file):
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge_base = f.read()
        logger.info(f"✅ Loaded knowledge base: {len(knowledge_base)} characters")
    else:
        logger.warning(f"⚠️  Knowledge file not found: {knowledge_file}")

    # ===== STT (Speech-to-Text) SERVICE SETUP =====
    stt_provider = os.getenv("STT_PROVIDER", "soniox").lower()
    
    if stt_provider == "soniox":
        # Build context for improved STT accuracy
        soniox_context = build_soniox_context(knowledge_base)
        
        stt = SonioxSTTService(
            api_key=os.getenv("SONIOX_API_KEY"),
            model=os.getenv("SONIOX_MODEL", "stt-rt-v3"),
            context=soniox_context  # Provide domain context for better accuracy
        )
        logger.info("✅ Soniox STT initialized with agricultural context")
    elif stt_provider == "deepgram":
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY")
        )
    elif stt_provider == "sarvam":
        stt = SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown STT provider: {stt_provider}")

    state = ConversationState()

    # ===== TTS (Text-to-Speech) SERVICE SETUP =====
    tts_provider = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
    
    if tts_provider == "murf":
        import aiohttp
        session = aiohttp.ClientSession()
        
        pronunciation_dict = {
            "DFNDR": {
                "type": "SAY_AS",
                "pronunciation": "Defender"
            }
        }
        
        tts = MurfTTSService(
            api_key=os.getenv("MURF_API_KEY"),
            params=MurfTTSService.InputParams(
                voice_id=os.getenv("MURF_VOICE_ID", "en-IN-ravi"),
                style=os.getenv("MURF_STYLE", "Conversational"),
                model=os.getenv("MURF_MODEL", "FALCON"),
                sample_rate=16000,
                format="PCM",
                channel_type="MONO",
                pronunciation_dictionary=pronunciation_dict
            )
        )
    elif tts_provider == "elevenlabs":
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVEN_LABS_API_KEY"),
            voice_id="pNInz6obpgDQGcFmaJgB",
            model="eleven_turbo_v2_5"
        )
    elif tts_provider == "cartesia":
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121"
        )
    elif tts_provider == "sarvam":
        tts = SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            target_language_code="te-IN",
            voice_id="manisha",
            model="bulbul:v2",
            enable_preprocessing=True,
            speech_sample_rate=22050,
            pitch=0,
            pace=1,
            loudness=1
        )
    elif tts_provider == "azure":
        tts = AzureTTSService(
            api_key=os.getenv("AZURE_API_KEY"),
            region=os.getenv("AZURE_REGION", "eastus"),
            voice=os.getenv("AZURE_VOICE", "te-IN-ShrutiNeural"),
            sample_rate=24000,
            params=AzureTTSService.InputParams(
                rate="1.05",
                pitch=None,
                style=None
            )
        )
    else:
        raise ValueError(f"Unknown TTS provider: {tts_provider}")

    # ===== LLM (Language Model) SERVICE SETUP =====
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if llm_provider == "azure":
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not azure_endpoint or not azure_api_key:
            raise ValueError("Azure OpenAI selected but credentials not set")
        
        base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{azure_deployment}"
        
        logger.info(f"🔷 Using Azure OpenAI - Deployment: {azure_deployment}")
        llm = AzureOpenAILLMService(
            model=azure_deployment,
            api_key=azure_api_key,
            base_url=base_url,
            stream=True,
            azure_api_version=azure_api_version
        )
    
    elif llm_provider == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        
        if not gemini_api_key:
            raise ValueError("Gemini selected but GEMINI_API_KEY not set")
        
        logger.info(f"🔷 Using Google Gemini - Model: {gemini_model}")
        llm = GoogleLLMService(
            model=gemini_model,
            api_key=gemini_api_key
        )
    
    elif llm_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        if not openai_api_key:
            raise ValueError("OpenAI selected but OPENAI_API_KEY not set")
        
        logger.info(f"🟢 Using OpenAI - Model: {openai_model}")
        llm = OpenAILLMService(
            model=openai_model,
            api_key=openai_api_key,
            stream=True
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}. Supported: 'openai', 'azure', 'gemini'")



    # ===== RAG SYSTEM INITIALIZATION =====
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY required for RAG embeddings")
        raise ValueError("OPENAI_API_KEY not set")
    
    rag_system = SimpleRAG(knowledge_base, openai_api_key)



    # ===== SYSTEM PROMPT CONFIGURATION ===== (RAG injects relevant chunks)
    system_prompt ="""You are an Biofactor Scientist.
    Respond ONLY in natural spoken Telugu used by farmers.
    Avoid formal Telugu.
    Avoid literal translations.
    Use verbs commonly spoken in villages. Never use English or any other language in your responses.
    
    Welcome message = "బయో ఫ్యాక్టరీ కి స్వాగతం. నేను మీ బయో ఫ్యాక్టర్ సైంటిస్ట్. మన కొత్త ప్రోడక్ట్ బయోల్మిన్ గురించి ఎటువంటి వివరాలు కావాలన్నా తెలియజేయగలరు"

    Conversation flow (strict rules):
    1) When a call connects, greet the user in Telugu using the welcome message and introduce yourself briefly.
    2) Keep replies short, simple and farmer-friendly. Do not provide answers to substantive questions until the user gives name and phone number.

    IMPORTANT:
    - Aim for 15 to 30 words per response.
    - Never exceed 35 words unless the user asks for more.
    - Answer questions ONLY based on the following knowledge base document.
    - Use previous user messages in this conversation to understand context, intent, and continuity.
    - Do NOT introduce information that is not present in the knowledge base.

    Conversational Memory & Context:
    - Treat this as an ongoing conversation, not a single question
    - Remember and use past user messages to understand what the user is referring to
    - If the user asks a follow-up question, connect it to previous questions naturally
    - Do NOT ask the user to repeat information already provided earlier

    Clarification Behavior:
    - If the user's question is incomplete, vague, or depends on missing details, politely ask a follow-up question in Telugu
    - Ask only what is necessary to continue the conversation
    - Do not guess or hallucinate missing information

    Knowledge Base (relevant context provided per query):
    {context}

    Rules you must strictly follow:
    1. Respond ONLY in pure Telugu (Unicode Telugu script).
    2. Do NOT use any English words, letters, numbers, symbols, emojis, or bullet points.
    3. Use natural spoken Telugu as used in phone conversations.
    4. Keep sentences short and clear.
    5. Use simple farmer-friendly agricultural language.
    6. Always include proper punctuation like commas, full stops, and question marks.
    7. If numbers or quantities are needed, write them fully in Telugu words.
    8. Avoid headings, lists, or markdown formatting.

    Response Rules:
    - Always respond in Telugu only
    - Be conversational, polite, and helpful
    - Answer strictly from the knowledge base
    - If the information is not found in the document, clearly say you do not have that information (in Telugu)
    - If clarification is required, ask a question instead of answering
    - If the knowledge base contains non-Telugu words or phrases, rewrite them in Telugu without changing the meaning
    """.format(context="[Context will be added per query]")

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    # ===== PIPELINE SETUP =====
    # Create RAG context injector and TTS preprocessor
    rag_injector = RAGContextInjector(rag_system, openai_api_key, messages)
    tts_preprocessor = TTSPreprocessor()

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            rag_injector,
            context_aggregator.user(),
            llm,
            tts_preprocessor,  # Preprocess text before TTS
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


# ===== MAIN BOT ENTRY POINT =====

async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    # ---------------- START PIPECAT BOT ----------------
    # Pipecat's main() will handle all endpoints including:
    # - / (health check)
    # - /client (playground)
    # - /api/offer, /start, /sessions/*, etc.
    import sys
    from pipecat.runner.run import main
    
    # Configure Pipecat to listen on 0.0.0.0 for Render deployment
    port = os.environ.get("PORT", "7860")
    
    print(f"\n🚀 Bot starting! Open this URL in your browser:\n👉 http://localhost:{port}\n")

    sys.argv.extend([
        "--transport", "webrtc",
        "--host", "127.0.0.1",
        "--port", port
    ])
    
    main()
