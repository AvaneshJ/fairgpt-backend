from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool # 🟢 IMPORT THREADPOOL
from pydantic import BaseModel
import os
from google import genai
from google.genai import types
from rag_engine import generate_hybrid_rag_news 
import asyncio
from fastapi.responses import JSONResponse

app = FastAPI(title="TruthLens Unbiased News API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client for Vision
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class NewsQuery(BaseModel):
    query: str
    language: str="English"

@app.post("/api/search")
async def search_news(data: NewsQuery):
    try:
        api_key = os.getenv("API_KEY")
        lang = getattr(data, "language", "English") 
        
        # 🟢 FIX: Run the heavy RAG engine in a background thread
        result = await run_in_threadpool(generate_hybrid_rag_news, data.query, api_key, lang)
        return result
        
    except Exception as e:
        print(f"🔥 SEARCH ERROR: {str(e)}") 
        raise HTTPException(
            status_code=500, 
            detail="The verification engine is currently experiencing high traffic. Please try again in a moment."
        )

@app.post("/api/verify-media")
async def verify_media(file: UploadFile = File(...), query: str = Form(None)):
    try:
        return await asyncio.wait_for(process_media_logic(file, query), timeout=60.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI processing took too long. Please try a simpler image.")
    except Exception as e:
        print(f"🔥 MEDIA ERROR: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Media verification failed. Our AI providers might be at capacity. Please try again."
        )

async def process_media_logic(file, user_query: str = None):
    await file.seek(0)
    file_bytes = await file.read()
    
    # 🟢 FIX: Added the faster Lite model to the top of the queue
    models_to_try = [
        "gemini-3.1-flash-lite-preview", 
        "gemini-3-flash-preview", 
    ]
    
    base_prompt = (
        "You are an expert news analyst. Read the text in this image. "
        "Identify the main news claim. Output ONLY a 1-sentence search query. "
        "CRITICAL: The entire response MUST be under 300 characters."
    )
    
    if user_query:
        extraction_prompt = f"{base_prompt} The user also added this context: '{user_query}'. Incorporate both into your query."
    else:
        extraction_prompt = base_prompt
    
    for model_name in models_to_try:
        try:
            print(f"🤖 Scanning with {model_name}...")
            # Note: Generating content via SDK can also be blocking. If it's slow, consider threadpooling it too!
            response = await run_in_threadpool(
                client.models.generate_content,
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=file.content_type),
                    extraction_prompt
                ]
            )
            extracted_query = response.text.strip()
            
            if extracted_query:
                print(f"✅ Extracted AI Query: {extracted_query}")
                
                # 🟢 FIX: Run the heavy RAG engine in a background thread
                verification_data = await run_in_threadpool(
                    generate_hybrid_rag_news, 
                    extracted_query, 
                    os.getenv("API_KEY"), 
                    "English"
                )
                verification_data["extractedQuery"] = extracted_query
                return verification_data
                
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ {model_name} Quota Full, trying next...")
                continue
            if "404" in str(e):
                print(f"⚠️ {model_name} Retired or Not Found.")
                continue
            raise e
            
    return {"status": "FAIL", "summary": "All models exhausted."}

@app.get("/")
def home():
    return {"message": "FairGPT Backend is Live!"}

@app.get("/api/health")
async def health_check():
    return JSONResponse(content={"status": "active", "message": "TruthLens backend is awake!"}, status_code=200)