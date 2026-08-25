from pathlib import Path
from dotenv import load_dotenv

# Always load backend/.env (not cwd-dependent); override empty shell vars
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from collections import defaultdict
from time import time
import os
import asyncio
import json
from google import genai
from google.genai import types
from rag_engine import generate_hybrid_rag_news, iter_hybrid_rag_news_events

app = FastAPI(title="TruthLens Unbiased News API")

# --- CONFIG ---
MAX_QUERY_LEN = 500
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"}
ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "video/webm",
}
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW = 60  # seconds

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
    ).split(",")
    if o.strip()
]
MAX_CONTEXT_LEN = 1800

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Verification engine is not configured.")
    return key


# Initialize Gemini Client for Vision (lazy-safe if key missing at import)
_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
client = genai.Client(api_key=_gemini_key) if _gemini_key else None

# --- RATE LIMITING (in-memory per IP) ---
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time()
    bucket = _rate_buckets[ip]
    _rate_buckets[ip] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_buckets[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")
    _rate_buckets[ip].append(now)


def require_api_key(request: Request) -> None:
    expected = os.getenv("BACKEND_API_KEY")
    if not expected:
        # Misconfigured server: refuse protected routes rather than leaving them open
        raise HTTPException(status_code=503, detail="API authentication is not configured.")
    provided = request.headers.get("X-API-Key") or ""
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        provided = provided or auth[7:].strip()
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class NewsQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    language: str = Field(default="English", max_length=64)
    context: str = Field(default="", max_length=MAX_CONTEXT_LEN)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        return cleaned

    @field_validator("context")
    @classmethod
    def strip_context(cls, v: str) -> str:
        return (v or "").strip()[:MAX_CONTEXT_LEN]


@app.post("/api/search")
async def search_news(
    data: NewsQuery,
    request: Request,
    _: None = Depends(require_api_key),
    __: None = Depends(enforce_rate_limit),
):
    try:
        api_key = get_gemini_api_key()
        result = await run_in_threadpool(
            generate_hybrid_rag_news,
            data.query,
            api_key,
            data.language,
            data.context or None,
        )
        if not isinstance(result, dict):
            return {"status": "FAIL", "summary": "Verification returned an unexpected response."}
        if result.get("status") == "FAIL":
            return JSONResponse(content=result, status_code=200)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"SEARCH ERROR: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="The verification engine is currently experiencing high traffic. Please try again in a moment.",
        )


def _sse_pack(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/search/stream")
async def search_news_stream(
    data: NewsQuery,
    request: Request,
    _: None = Depends(require_api_key),
    __: None = Depends(enforce_rate_limit),
):
    """SSE transport for progressive generation. Final `result` event keeps core JSON schema."""
    api_key = get_gemini_api_key()

    def event_iter():
        try:
            for evt in iter_hybrid_rag_news_events(
                data.query,
                api_key,
                data.language,
                data.context or None,
            ):
                yield _sse_pack(evt)
        except Exception as e:
            print(f"STREAM SEARCH ERROR: {str(e)}")
            yield _sse_pack(
                {
                    "event": "result",
                    "data": {
                        "status": "FAIL",
                        "verdict": "Unclear",
                        "summary": "Audit error: The AI models are currently experiencing heavy traffic. Please wait a moment and try again.",
                        "counter_summary": "",
                        "clarifications": [],
                        "audit_history": [],
                        "logic_audit": "",
                        "certainty": 0,
                        "evidence_timeline": [],
                        "verification_audit": {
                            "goldenCount": 0,
                            "consensusCount": 0,
                            "rawCount": 0,
                        },
                        "bias_score": 0,
                        "bias_reason": "",
                        "sources": [],
                    },
                }
            )

    return StreamingResponse(
        event_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/verify-media")
async def verify_media(
    request: Request,
    file: UploadFile = File(...),
    query: str = Form(None),
    language: str = Form("English"),
    _: None = Depends(require_api_key),
    __: None = Depends(enforce_rate_limit),
):
    try:
        return await asyncio.wait_for(
            process_media_logic(file, query, language), timeout=60.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail="AI processing took too long. Please try a simpler image."
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"MEDIA ERROR: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Media verification failed. Our AI providers might be at capacity. Please try again.",
        )


async def process_media_logic(file: UploadFile, user_query: str = None, language: str = "English"):
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPEG, PNG, WebP, or GIF image.",
        )

    await file.seek(0)
    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image must be 5MB or smaller.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if user_query:
        user_query = user_query.strip()[:MAX_QUERY_LEN]
    language = (language or "English").strip()[:64] or "English"

    if client is None:
        raise HTTPException(status_code=503, detail="Verification engine is not configured.")

    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]

    base_prompt = (
        "You are an expert news analyst reading a screenshot or share-card image.\n"
        "1) Transcribe visible headline/body text accurately (OCR).\n"
        "2) Identify the single main verifiable news claim.\n"
        "3) Produce a concise web search query for fact-checking.\n"
        "Return ONLY valid JSON with keys:\n"
        '{"ocr_text":"...","primary_claim":"...","search_query":"..."}\n'
        "Rules: no markdown, search_query under 280 characters, prefer India context unless another country is explicit."
    )

    if user_query:
        # Avoid raw quote injection into the prompt body
        safe_ctx = user_query.replace("'", "")[:300]
        extraction_prompt = (
            f"{base_prompt}\nUser-provided context to incorporate: {safe_ctx}."
        )
    else:
        extraction_prompt = base_prompt

    for model_name in models_to_try:
        try:
            print(f"Scanning with {model_name}...")
            response = await run_in_threadpool(
                client.models.generate_content,
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=content_type),
                    extraction_prompt,
                ],
            )
            raw_extract = (response.text or "").strip()
            extracted_query = ""
            ocr_text = ""
            primary_claim = ""

            try:
                cleaned = raw_extract.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    ocr_text = str(parsed.get("ocr_text") or "").strip()[:2000]
                    primary_claim = str(parsed.get("primary_claim") or "").strip()[:500]
                    extracted_query = str(
                        parsed.get("search_query") or primary_claim or ""
                    ).strip()[:MAX_QUERY_LEN]
            except Exception:
                extracted_query = raw_extract[:MAX_QUERY_LEN]
                primary_claim = extracted_query

            if not extracted_query and primary_claim:
                extracted_query = primary_claim[:MAX_QUERY_LEN]
            if not extracted_query and ocr_text:
                extracted_query = ocr_text[:MAX_QUERY_LEN]

            if extracted_query:
                print(f"Extracted AI Query: {extracted_query}")
                verification_data = await run_in_threadpool(
                    generate_hybrid_rag_news,
                    extracted_query,
                    get_gemini_api_key(),
                    language,
                )
                if not isinstance(verification_data, dict):
                    return {
                        "status": "FAIL",
                        "summary": "Verification returned an unexpected response.",
                        "extractedQuery": extracted_query,
                    }
                verification_data["extractedQuery"] = extracted_query
                # Optional UI helpers — core SUCCESS fields unchanged
                if ocr_text:
                    verification_data["extractedText"] = ocr_text
                if primary_claim:
                    verification_data["primaryClaim"] = primary_claim
                return verification_data

        except HTTPException:
            raise
        except Exception as e:
            err = str(e)
            if "429" in err:
                print(f"{model_name} Quota Full, trying next...")
                continue
            if "404" in err:
                print(f"{model_name} Retired or Not Found.")
                continue
            raise e

    return {"status": "FAIL", "summary": "All models exhausted."}


@app.post("/api/verify-audio")
async def verify_audio(
    request: Request,
    file: UploadFile = File(...),
    query: str = Form(None),
    language: str = Form("English"),
    _: None = Depends(require_api_key),
    __: None = Depends(enforce_rate_limit),
):
    try:
        return await asyncio.wait_for(
            process_audio_logic(file, query, language), timeout=90.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Audio processing took too long. Please try a shorter clip.",
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"AUDIO ERROR: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Audio verification failed. Please try again.",
        )


async def process_audio_logic(file: UploadFile, user_query: str = None, language: str = "English"):
    content_type = (file.content_type or "").lower().split(";")[0].strip() or "audio/webm"
    if content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio type. Upload MP3, WAV, M4A, WEBM, OGG, or FLAC.",
        )

    await file.seek(0)
    file_bytes = await file.read(MAX_AUDIO_BYTES + 1)
    if len(file_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="Audio must be 10MB or smaller.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded.")

    if user_query:
        user_query = user_query.strip()[:MAX_QUERY_LEN]
    language = (language or "English").strip()[:64] or "English"

    if client is None:
        raise HTTPException(status_code=503, detail="Verification engine is not configured.")

    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]

    base_prompt = (
        "You are an expert news analyst listening to an audio clip (voice note, news clip, or speech).\n"
        "1) Transcribe the spoken content accurately.\n"
        "2) Identify the single main verifiable news claim.\n"
        "3) Produce a concise web search query for fact-checking.\n"
        "Return ONLY valid JSON with keys:\n"
        '{"transcript":"...","primary_claim":"...","search_query":"..."}\n'
        "Rules: no markdown, search_query under 280 characters, prefer India context unless another country is explicit."
    )
    if user_query:
        safe_ctx = user_query.replace("'", "")[:300]
        extraction_prompt = f"{base_prompt}\nUser-provided context to incorporate: {safe_ctx}."
    else:
        extraction_prompt = base_prompt

    # Normalize mime for Gemini (video/webm recordings are still audio)
    mime_for_model = "audio/webm" if content_type == "video/webm" else content_type
    if mime_for_model == "audio/mp3":
        mime_for_model = "audio/mpeg"

    for model_name in models_to_try:
        try:
            print(f"Listening with {model_name}...")
            response = await run_in_threadpool(
                client.models.generate_content,
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_for_model),
                    extraction_prompt,
                ],
            )
            raw_extract = (response.text or "").strip()
            extracted_query = ""
            transcript = ""
            primary_claim = ""

            try:
                cleaned = raw_extract.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    transcript = str(parsed.get("transcript") or "").strip()[:3000]
                    primary_claim = str(parsed.get("primary_claim") or "").strip()[:500]
                    extracted_query = str(
                        parsed.get("search_query") or primary_claim or ""
                    ).strip()[:MAX_QUERY_LEN]
            except Exception:
                extracted_query = raw_extract[:MAX_QUERY_LEN]
                transcript = extracted_query
                primary_claim = extracted_query

            if not extracted_query and primary_claim:
                extracted_query = primary_claim[:MAX_QUERY_LEN]
            if not extracted_query and transcript:
                extracted_query = transcript[:MAX_QUERY_LEN]

            if extracted_query:
                print(f"Extracted audio query: {extracted_query}")
                verification_data = await run_in_threadpool(
                    generate_hybrid_rag_news,
                    extracted_query,
                    get_gemini_api_key(),
                    language,
                )
                if not isinstance(verification_data, dict):
                    return {
                        "status": "FAIL",
                        "summary": "Verification returned an unexpected response.",
                        "extractedQuery": extracted_query,
                    }
                verification_data["extractedQuery"] = extracted_query
                if transcript:
                    verification_data["extractedText"] = transcript
                if primary_claim:
                    verification_data["primaryClaim"] = primary_claim
                verification_data["inputType"] = "audio"
                return verification_data

        except HTTPException:
            raise
        except Exception as e:
            err = str(e)
            if "429" in err:
                print(f"{model_name} Quota Full, trying next...")
                continue
            if "404" in err:
                print(f"{model_name} Retired or Not Found.")
                continue
            raise e

    return {"status": "FAIL", "summary": "All models exhausted."}


@app.get("/")
def home():
    return {"message": "TruthLens Backend is Live!"}


@app.get("/api/health")
async def health_check():
    return JSONResponse(
        content={"status": "active", "message": "TruthLens backend is awake!"},
        status_code=200,
    )
