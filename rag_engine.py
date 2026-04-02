import os
import requests
import re
import datetime
from firebase_admin import firestore
from difflib import SequenceMatcher
from tavily import TavilyClient 
from source_reputation import get_source_profile
from config import API_URL_BASE, MODEL_NAME, SYSTEM_INSTRUCTION, RAW_NEWS_COLLECTION
from database_setup import DB
import json
import hashlib
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- HELPER FUNCTIONS ---

def parse_ai_response(text):
    # List of tags we expect in order
    tags = ["[SUMMARY]", "[COUNTER_SUMMARY]", "[CLARIFICATION]", "[AUDIT]", "[LOGIC_AUDIT]", "[CONFIDENCE]", "[TIMELINE]","[BIAS_METER]", "[BIAS_REASON]"]
    sections = {}
    
    # Map where every tag starts in the raw text
    positions = []
    for tag in tags:
        idx = text.find(tag)
        if idx != -1:
            positions.append((idx, tag))
    
    # Sort positions by where they appear in the string
    positions.sort()
    
    # Slice the string between the tags
    for i in range(len(positions)):
        start_idx, current_tag = positions[i]
        start_content = start_idx + len(current_tag)
        
        # If there is a next tag, end there; otherwise, go to the end of the string
        if i + 1 < len(positions):
            end_idx = positions[i+1][0]
            sections[current_tag] = text[start_content:end_idx].strip()
        else:
            sections[current_tag] = text[start_content:].strip()
            
    return sections

def filter_relevant_sources(core_claim: str, search_results: list, max_to_keep: int = 5) -> list:
    """Scores and filters search results to remove irrelevant SEO spam."""
    if not search_results: return []
    
    scored_results = []
    # Strip punctuation and get core words
    claim_tokens = set(re.findall(r'\w+', core_claim.lower())) 
    
    for res in search_results:
        text_to_search = (res.get('title', '') + " " + res.get('content', '')).lower()
        text_tokens = set(re.findall(r'\w+', text_to_search))
        
        # 1. Keyword Overlap: How many words from the claim are in the article?
        match_score = len(claim_tokens.intersection(text_tokens)) / max(1, len(claim_tokens))
        
        # 2. Phrasing Overlap: Does it use similar sentence structures?
        seq_score = SequenceMatcher(None, core_claim.lower(), text_to_search).ratio()
        
        # Combine scores
        final_score = match_score + (seq_score * 2) 
        
        # Only keep articles that pass the minimum relevance threshold
        if final_score > 0.05: 
            res['relevance_score'] = final_score
            scored_results.append(res)
            
    # Sort by highest score first, then slice to keep only the absolute best ones
    scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return scored_results[:max_to_keep]


# --- AGENTIC PLANNER ---

def generate_search_plan(user_query: str, api_key: str) -> list:
    """Agentic step: Breaks a single claim into a targeted 3-part search strategy."""
    print(f"\n🧠 Planning investigation for: {user_query}")
    
    planning_prompt = f"""
    You are an investigative journalist planning a fact-check for the following claim:
    CLAIM: "{user_query}"
    
    Create a search strategy to verify this. Output ONLY a raw JSON list of exactly 3 search string queries. Do not include markdown formatting.
    1. A query to find the primary factual event (Who, What, Where).
    2. A query to find official government or institutional responses.
    3. A query to find criticisms, debunkings, or alternative context.
    
    Example Output format:
    ["exact quote or event search", "official statement on event", "criticism or debunking of event"]
    CRITICAL RULES FOR ANALYSIS:
    1. GEO-DEFAULT: Unless the user explicitly names a foreign country, you MUST assume the context is India.
    2. Check claims against official Indian sources like the RBI, PIB, or established Indian news outlets before analyzing.
    """
    
    
    payload = {
        "contents": [{"parts": [{"text": planning_prompt}]}],
    }
    
    try:
        response = requests.post(f"{API_URL_BASE}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}", json=payload, timeout=20)
        response.raise_for_status()
        raw_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        
        # Clean the output (in case the AI adds markdown blocks)
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        search_plan = json.loads(clean_text)
        print(f"📋 Plan generated: {search_plan}")
        return search_plan
    except Exception as e:
        print(f"⚠️ Planner failed, falling back to raw query. Error: {e}")
        # Fallback to just using the user's original query 3 times
        return [user_query, user_query, user_query]


# --- CORE ORCHESTRATION ---

def generate_hybrid_rag_news(user_query: str, api_key: str, language: str="English"):
    try:
        print(f"\n🔍 --- AUDIT START: {user_query} ---")
        
        # 1. CALL THE PLANNING AGENT
        search_plan = generate_search_plan(user_query, api_key)
        
        # Safely unpack the queries
        q_factual = search_plan[0] if len(search_plan) > 0 else user_query 
        q_official = search_plan[1] if len(search_plan) > 1 else user_query
        q_alt = search_plan[2] if len(search_plan) > 2 else f'criticism of "{user_query}" OR "opposition to {user_query}"'

        GOLDEN_LIST = [
    # Core Government & Constitutional Bodies
    "pib.gov.in", "india.gov.in", "mp.gov.in", "rbi.org.in", 
    "eci.gov.in", "mohfw.gov.in", "uidai.gov.in", "mea.gov.in","india"
    # Global Authorities
    "who.int", "un.org", "worldbank.org",
    # Certified Fact-Checking Units
    "boomlive.in", "factly.in", "altnews.in", "newschecker.in", 
    "vishvasnews.com", "logically.ai", "smhoaxdetect.com"
]
        CONSENSUS_LIST = [
    # State Media & News Wires
    "ddnews.gov.in", "newsonair.gov.in", "ptinews.com", "aniin.com",
    # Mainstream Newspapers of Record
    "thehindu.com", "indianexpress.com", "theprint.in", "livemint.com",
    # International Wires
    "reuters.com", "apnews.com", "bbc.com/news", "bloomberg.com", "aljazeera.com"
]
        
        # 2. EXECUTE TARGETED SEARCHES
        print(f"🔎 Executing Agentic Searches...")
        print(f"   -> Golden Query: {q_official}")
        print(f"   -> Consensus Query: {q_factual}")
        print(f"   -> Alt Query: {q_alt}")
        
        g_res = tavily.search(
            query=q_official, 
            include_domains=GOLDEN_LIST, 
            search_depth="advanced", 
            max_results=4
        )
        c_res = tavily.search(
            query=q_factual, 
            include_domains=CONSENSUS_LIST, 
            search_depth="advanced", 
            max_results=3
        )
        alt_res = tavily.search(
            query=q_alt, 
            search_depth="advanced", 
            max_results=3
        )

        # 🟢 3. SEMANTIC RE-RANKING (FILTERING NOISE)
        print("🧹 Re-ranking and filtering SEO noise...")
        raw_consensus = g_res.get('results', []) + c_res.get('results', [])
        raw_alternative = alt_res.get('results', [])
        
        # Scrub the lists to keep only the highly relevant context
        filtered_consensus = filter_relevant_sources(user_query, raw_consensus, max_to_keep=6)
        filtered_alternative = filter_relevant_sources(user_query, raw_alternative, max_to_keep=3)

        # Build context strings securely using the filtered arrays
        consensus_context = "\n\n".join([f"SOURCE: {r.get('url')}\nPUBLISHED: {r.get('published_date', 'Date Unknown')}\n{r.get('content')}" for r in filtered_consensus])
        alternative_context = "\n\n".join([f"SOURCE: {r.get('url')}\nPUBLISHED: {r.get('published_date', 'Date Unknown')}\n{r.get('content')}" for r in filtered_alternative])

        # 4. STRICT MASTER INTEGRITY SCAN
        # Ensure the integrity scan also only uses the filtered, relevant list
        all_results = filtered_consensus + filtered_alternative
        unique_urls = list(set([r.get('url') for r in all_results if r.get('url')]))
        
        counts = {"gold": 0, "con": 0, "raw": 0}
        verified_sources = []

        for url in unique_urls:
            low_url = url.lower()
            profile = get_source_profile(url)
            
            # Categorization logic
            if any(domain in low_url for domain in GOLDEN_LIST):
                counts["gold"] += 1
                verified_sources.append({"url": url, "meta": profile, "rank": 1})
            elif any(domain in low_url for domain in CONSENSUS_LIST):
                counts["con"] += 1
                verified_sources.append({"url": url, "meta": profile, "rank": 2})
            else:
                counts["raw"] += 1
                verified_sources.append({"url": url, "meta": profile, "rank": 3})

        # Sort sources so Golden and Consensus appear first in the UI
        verified_sources.sort(key=lambda x: x['rank'])

        # 5. AI GENERATION
        translation_rule=f"\n\nCRITICAL LANGUAGE INSTRUCTION: You MUST write the actual content for the summary, counter-summary, clarifications, and logic audit in {language}. However, you MUST keep the structural tags themselves EXACTLY in English (e.g., write '[SUMMARY] (Target language text...) [COUNTER_SUMMARY] (Target language text...)'). Do not translate the bracketed tags."
        payload = {
            "contents": [{"parts": [{"text": f"QUERY: {user_query}\n\nCONSENSUS:\n{consensus_context}\n\nALTERNATIVE:\n{alternative_context}"}]}],
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION + "\n\nSTRUCTURE: [SUMMARY], [COUNTER_SUMMARY], [CLARIFICATION], [AUDIT], [LOGIC_AUDIT], [CONFIDENCE]. Do not use markdown headers." + translation_rule}]}
        }
        
        response = requests.post(f"{API_URL_BASE}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}", json=payload, timeout=45)
        response.raise_for_status()
        raw_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        parsed = parse_ai_response(raw_text)

        # 6. STABILITY PARSER
        def get_tag_content(text, tag, next_tag=None):
            start = text.find(tag)
            if start == -1: return ""
            start += len(tag)
            if next_tag:
                end = text.find(next_tag, start)
                return text[start:end].strip() if end != -1 else text[start:].strip()
            return text[start:].strip()

        def to_list(s):
            if not s: return []
            return [line.strip("- ").strip() for line in s.splitlines() if line.strip()]

        summary = parsed.get("[SUMMARY]", "Summary unavailable.")
        counter = parsed.get("[COUNTER_SUMMARY]", "No alternative view found.")
        clari = to_list(parsed.get("[CLARIFICATION]", ""))
        audit_trail = to_list(parsed.get("[AUDIT]", ""))
        logic = parsed.get("[LOGIC_AUDIT]", "Audit complete.")
        conf_val = parsed.get("[CONFIDENCE]", "95")
        timeline=parsed.get("[TIMELINE]", "[]")
        raw_bias_score = parsed.get("[BIAS_METER]", "0").strip()
        bias_reason = parsed.get("[BIAS_REASON]", "No bias detected.")
        try:
            clean_timeline = timeline.replace("```json","").replace("```", "").strip()
            timeline_data = json.loads(clean_timeline) if clean_timeline else []
        except:
            timeline_data = []
        try:
            clean_bias_score=int(re.search(r'\d+', raw_bias_score).group())
            normalised_score=clean_bias_score/100.0
        except:
            normalised_score=0

        return {
            "status": "SUCCESS",
            "summary": summary or "Consensus summary verified.",
            "counter_summary": counter or "No significant alternative perspective found.",
            "clarifications": clari,
            "audit_history": audit_trail,
            "logic_audit": logic or "Audit complete.",
            "certainty": int(re.search(r'\d+', conf_val).group()) if re.search(r'\d+', conf_val) else 95,
            "evidence_timeline": timeline_data,
            "verification_audit": {"goldenCount": counts["gold"], "consensusCount": counts["con"], "rawCount": counts["raw"]},
            "bias_score": normalised_score,
            "bias_reason": bias_reason,
            "sources": verified_sources[:8]
        }

    # 🟢 THE SECURE WAY
    except Exception as e:
    # 1. Print the raw error to your secure Render console
        print(f"🔥 Internal Gemini API Error: {str(e)}") 
    
    # 2. Return a safe, hardcoded string to the frontend
        return "Audit error: The AI models are currently experiencing heavy traffic. Please wait a moment and try again."