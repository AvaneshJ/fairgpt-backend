import os
import sys
import requests
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from difflib import SequenceMatcher
from tavily import TavilyClient
from source_reputation import get_source_profile
from config import API_URL_BASE, MODEL_NAME, SYSTEM_INSTRUCTION
import json

# Windows consoles often use cp1252; emoji debug logs must not crash audits
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

CONTEXT_MAX_CHARS = 1800

GOLDEN_LIST = [
    # Core Government & Constitutional Bodies
    "pib.gov.in", "india.gov.in", "mp.gov.in", "rbi.org.in",
    "eci.gov.in", "mohfw.gov.in", "uidai.gov.in", "mea.gov.in",
    # Global Authorities
    "who.int", "un.org", "worldbank.org",
    # Certified Fact-Checking Units
    "boomlive.in", "factly.in", "altnews.in", "newschecker.in",
    "vishvasnews.com", "logically.ai", "smhoaxdetect.com",
]
CONSENSUS_LIST = [
    # State Media & News Wires
    "ddnews.gov.in", "newsonair.gov.in", "ptinews.com", "aniin.com",
    # Mainstream Newspapers of Record
    "thehindu.com", "indianexpress.com", "theprint.in", "livemint.com",
    # International Wires
    "reuters.com", "apnews.com", "bbc.com", "bloomberg.com", "aljazeera.com",
]

SYNTHESIS_STRUCTURE = (
    "STRUCTURE: [SUMMARY], [COUNTER_SUMMARY], [CLARIFICATION], [AUDIT], "
    "[LOGIC_AUDIT], [CONFIDENCE], [TIMELINE], [BIAS_METER], [BIAS_REASON]. "
    "Do not use markdown headers."
)


def hostname_matches_domain(url: str, domain: str) -> bool:
    """True if URL hostname equals domain or is a subdomain of it."""
    try:
        host = urlparse(url).hostname or ""
        host = host.lower().rstrip(".")
        domain = domain.lower().strip().rstrip("/")
        # Paths like bbc.com/news are treated as hostname + optional path prefix
        if "/" in domain:
            domain_host, domain_path = domain.split("/", 1)
            if not (host == domain_host or host.endswith("." + domain_host)):
                return False
            path = (urlparse(url).path or "").lower()
            return path.startswith("/" + domain_path) or path == "/" + domain_path
        return host == domain or host.endswith("." + domain)
    except Exception:
        return False


def derive_verdict(certainty: int, gold: int, consensus: int, bias_score: float) -> str:
    """Map confidence + source mix into a simple UI verdict."""
    if certainty >= 70 and (gold + consensus) >= 2 and bias_score <= 0.6:
        return "Supported"
    if certainty <= 40 or (gold + consensus) == 0:
        return "Disputed"
    return "Unclear"


def fail_result(summary: str) -> dict:
    """Standard FAIL dict shape used by all error paths."""
    return {
        "status": "FAIL",
        "verdict": "Unclear",
        "summary": summary,
        "counter_summary": "",
        "clarifications": [],
        "audit_history": [],
        "logic_audit": "",
        "certainty": 0,
        "evidence_timeline": [],
        "verification_audit": {"goldenCount": 0, "consensusCount": 0, "rawCount": 0},
        "bias_score": 0,
        "bias_reason": "",
        "sources": [],
    }


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def _truncate_context(context: str | None) -> str | None:
    if not context:
        return None
    text = context.strip()
    if not text:
        return None
    if len(text) > CONTEXT_MAX_CHARS:
        return text[:CONTEXT_MAX_CHARS]
    return text


def _conversation_context_block(context: str | None) -> str:
    ctx = _truncate_context(context)
    if not ctx:
        return ""
    return f"\n\nCONVERSATION_CONTEXT:\n{ctx}\n"


# --- HELPER FUNCTIONS ---

def parse_ai_response(text):
    # List of tags we expect in order
    tags = ["[SUMMARY]", "[COUNTER_SUMMARY]", "[CLARIFICATION]", "[AUDIT]", "[LOGIC_AUDIT]", "[CONFIDENCE]", "[TIMELINE]", "[BIAS_METER]", "[BIAS_REASON]"]
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
            end_idx = positions[i + 1][0]
            sections[current_tag] = text[start_content:end_idx].strip()
        else:
            sections[current_tag] = text[start_content:].strip()

    return sections


def filter_relevant_sources(core_claim: str, search_results: list, max_to_keep: int = 5) -> list:
    """Scores and filters search results to remove irrelevant SEO spam."""
    if not search_results:
        return []

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


def build_success_result(raw_text: str, counts: dict, verified_sources: list) -> dict:
    """Parse synthesis text and assemble the SUCCESS response dict."""
    parsed = parse_ai_response(raw_text)

    def to_list(s):
        if not s:
            return []
        return [line.strip("- ").strip() for line in s.splitlines() if line.strip()]

    summary = parsed.get("[SUMMARY]", "Summary unavailable.")
    counter = parsed.get("[COUNTER_SUMMARY]", "No alternative view found.")
    clari = to_list(parsed.get("[CLARIFICATION]", ""))
    audit_trail = to_list(parsed.get("[AUDIT]", ""))
    logic = parsed.get("[LOGIC_AUDIT]", "Audit complete.")
    conf_val = parsed.get("[CONFIDENCE]", "")
    timeline = parsed.get("[TIMELINE]", "[]")
    raw_bias_score = parsed.get("[BIAS_METER]", "0").strip()
    bias_reason = parsed.get("[BIAS_REASON]", "No bias detected.")
    try:
        clean_timeline = timeline.replace("```json", "").replace("```", "").strip()
        timeline_data = json.loads(clean_timeline) if clean_timeline else []
    except Exception:
        timeline_data = []
    try:
        clean_bias_score = int(re.search(r"\d+", raw_bias_score).group())
        normalised_score = clean_bias_score / 100.0
    except Exception:
        normalised_score = 0

    conf_match = re.search(r"\d+", conf_val or "")
    # Failed/missing parse must not overstate certainty
    certainty = int(conf_match.group()) if conf_match else 25
    certainty = max(0, min(100, certainty))
    verdict = derive_verdict(
        certainty, counts["gold"], counts["con"], normalised_score
    )

    return {
        "status": "SUCCESS",
        "verdict": verdict,
        "summary": summary or "Consensus summary verified.",
        "counter_summary": counter or "No significant alternative perspective found.",
        "clarifications": clari,
        "audit_history": audit_trail,
        "logic_audit": logic or "Audit complete.",
        "certainty": certainty,
        "evidence_timeline": timeline_data,
        "verification_audit": {
            "goldenCount": counts["gold"],
            "consensusCount": counts["con"],
            "rawCount": counts["raw"],
        },
        "bias_score": normalised_score,
        "bias_reason": bias_reason,
        "sources": verified_sources[:8],
    }


# --- AGENTIC PLANNER ---

def generate_search_plan(user_query: str, api_key: str, context: str | None = None) -> list:
    """Agentic step: Breaks a single claim into a targeted 3-part search strategy."""
    _safe_print(f"\n[plan] Planning investigation for: {user_query}")

    ctx_block = _conversation_context_block(context)
    planning_prompt = f"""
    You are an investigative journalist planning a fact-check for the following claim:
    CLAIM: "{user_query}"
    {ctx_block}
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
        response = requests.post(
            f"{API_URL_BASE}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        raw_text = response.json()['candidates'][0]['content']['parts'][0]['text']

        # Clean the output (in case the AI adds markdown blocks)
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()

        search_plan = json.loads(clean_text)
        _safe_print(f"[plan] Plan generated: {search_plan}")
        return search_plan
    except Exception as e:
        _safe_print(f"[plan] Planner failed, falling back to raw query. Error: {e}")
        # Fallback to just using the user's original query 3 times
        return [user_query, user_query, user_query]


# --- PIPELINE STEPS ---

def unpack_search_plan(search_plan: list, user_query: str) -> tuple:
    """Safely unpack the 3 planned queries."""
    q_factual = search_plan[0] if len(search_plan) > 0 else user_query
    q_official = search_plan[1] if len(search_plan) > 1 else user_query
    q_alt = search_plan[2] if len(search_plan) > 2 else f'criticism of "{user_query}" OR "opposition to {user_query}"'
    return q_factual, q_official, q_alt


def run_parallel_searches(q_official: str, q_factual: str, q_alt: str) -> tuple:
    """Run golden/official, consensus/factual, and alternative Tavily searches in parallel."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise RuntimeError("TAVILY_API_KEY is missing. Cannot run evidence searches.")

    _safe_print("[search] Executing Agentic Searches (parallel)...")
    _safe_print(f"   -> Golden Query: {q_official}")
    _safe_print(f"   -> Consensus Query: {q_factual}")
    _safe_print(f"   -> Alt Query: {q_alt}")

    def _golden():
        return tavily.search(
            query=q_official,
            include_domains=GOLDEN_LIST,
            search_depth="advanced",
            max_results=4,
        )

    def _consensus():
        return tavily.search(
            query=q_factual,
            include_domains=CONSENSUS_LIST,
            search_depth="advanced",
            max_results=3,
        )

    def _alternative():
        return tavily.search(
            query=q_alt,
            search_depth="advanced",
            max_results=3,
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_g = executor.submit(_golden)
        fut_c = executor.submit(_consensus)
        fut_a = executor.submit(_alternative)
        g_res = fut_g.result()
        c_res = fut_c.result()
        alt_res = fut_a.result()

    return g_res, c_res, alt_res


def filter_search_results(user_query: str, g_res: dict, c_res: dict, alt_res: dict) -> tuple:
    """Semantic re-ranking: filter SEO noise from search results."""
    _safe_print("[filter] Re-ranking and filtering SEO noise...")
    raw_consensus = g_res.get('results', []) + c_res.get('results', [])
    raw_alternative = alt_res.get('results', [])

    filtered_consensus = filter_relevant_sources(user_query, raw_consensus, max_to_keep=6)
    filtered_alternative = filter_relevant_sources(user_query, raw_alternative, max_to_keep=3)
    return filtered_consensus, filtered_alternative


def build_source_contexts(filtered_consensus: list, filtered_alternative: list) -> tuple:
    """Build context strings and verified source list from filtered results."""
    consensus_context = "\n\n".join([
        f"SOURCE: {r.get('url')}\nPUBLISHED: {r.get('published_date', 'Date Unknown')}\n{r.get('content')}"
        for r in filtered_consensus
    ])
    alternative_context = "\n\n".join([
        f"SOURCE: {r.get('url')}\nPUBLISHED: {r.get('published_date', 'Date Unknown')}\n{r.get('content')}"
        for r in filtered_alternative
    ])

    all_results = filtered_consensus + filtered_alternative
    unique_urls = list(set([r.get('url') for r in all_results if r.get('url')]))

    counts = {"gold": 0, "con": 0, "raw": 0}
    verified_sources = []

    for url in unique_urls:
        profile = get_source_profile(url)

        if any(hostname_matches_domain(url, domain) for domain in GOLDEN_LIST):
            counts["gold"] += 1
            verified_sources.append({"url": url, "meta": profile, "rank": 1})
        elif any(hostname_matches_domain(url, domain) for domain in CONSENSUS_LIST):
            counts["con"] += 1
            verified_sources.append({"url": url, "meta": profile, "rank": 2})
        else:
            counts["raw"] += 1
            verified_sources.append({"url": url, "meta": profile, "rank": 3})

    verified_sources.sort(key=lambda x: x['rank'])
    return consensus_context, alternative_context, counts, verified_sources


def build_synthesis_payload(
    user_query: str,
    consensus_context: str,
    alternative_context: str,
    language: str,
    context: str | None = None,
) -> dict:
    """Build Gemini generateContent / streamGenerateContent request body."""
    translation_rule = (
        f"\n\nCRITICAL LANGUAGE INSTRUCTION: You MUST write the actual content for the summary, "
        f"counter-summary, clarifications, and logic audit in {language}. However, you MUST keep "
        f"the structural tags themselves EXACTLY in English (e.g., write '[SUMMARY] (Target language "
        f"text...) [COUNTER_SUMMARY] (Target language text...)'). Do not translate the bracketed tags."
    )
    ctx_block = _conversation_context_block(context)
    user_text = (
        f"QUERY: {user_query}"
        f"{ctx_block}\n"
        f"CONSENSUS:\n{consensus_context}\n\n"
        f"ALTERNATIVE:\n{alternative_context}"
    )
    return {
        "contents": [{"parts": [{"text": user_text}]}],
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION + "\n\n" + SYNTHESIS_STRUCTURE + translation_rule}]
        },
    }


def synthesize_non_stream(payload: dict, api_key: str) -> str:
    """Non-streaming Gemini synthesis; returns full raw text."""
    response = requests.post(
        f"{API_URL_BASE}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}",
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    return response.json()['candidates'][0]['content']['parts'][0]['text']


def iter_synthesize_stream(payload: dict, api_key: str):
    """
    Stream Gemini synthesis via SSE.
    Yields ("token", new_suffix) then ("done", full_raw_text).
    Handles both cumulative and incremental part text.
    """
    url = (
        f"{API_URL_BASE}/v1beta/models/{MODEL_NAME}:streamGenerateContent"
        f"?alt=sse&key={api_key}"
    )
    accumulated = ""
    prev_len = 0

    with requests.post(url, json=payload, timeout=120, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            try:
                piece = chunk["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                continue
            if not piece:
                continue

            # Gemini may send cumulative or incremental text
            if piece.startswith(accumulated) and len(piece) >= len(accumulated):
                # Cumulative: full text so far
                new_suffix = piece[prev_len:]
                accumulated = piece
                prev_len = len(accumulated)
            else:
                # Incremental: append delta
                accumulated += piece
                new_suffix = piece
                prev_len = len(accumulated)

            if new_suffix:
                yield ("token", new_suffix)

    yield ("done", accumulated)


def handle_pipeline_error(e: Exception) -> dict:
    """Shared FAIL handler with Windows-safe logging."""
    err = str(e)
    try:
        print(f"Internal Gemini API Error: {err}")
    except UnicodeEncodeError:
        print(f"Internal Gemini API Error: {err.encode('ascii', 'replace').decode('ascii')}")
    if isinstance(e, UnicodeEncodeError) or "codec can't encode" in err:
        summary = "Audit error: Local logging failed on this machine encoding. Please retry."
    elif "TAVILY_API_KEY" in err:
        summary = "Audit error: Search API key is missing. Configure TAVILY_API_KEY and retry."
    else:
        summary = (
            "Audit error: The AI models are currently experiencing heavy traffic. "
            "Please wait a moment and try again."
        )
    return fail_result(summary)


def _prepare_evidence(user_query: str, api_key: str, context: str | None = None) -> dict:
    """
    Shared pre-synthesis pipeline: plan -> parallel search -> filter -> sources.
    Returns dict with consensus_context, alternative_context, counts, verified_sources.
    Raises on hard failures (e.g. missing Tavily key).
    """
    search_plan = generate_search_plan(user_query, api_key, context=context)
    q_factual, q_official, q_alt = unpack_search_plan(search_plan, user_query)
    g_res, c_res, alt_res = run_parallel_searches(q_official, q_factual, q_alt)
    filtered_consensus, filtered_alternative = filter_search_results(
        user_query, g_res, c_res, alt_res
    )
    consensus_context, alternative_context, counts, verified_sources = build_source_contexts(
        filtered_consensus, filtered_alternative
    )
    return {
        "consensus_context": consensus_context,
        "alternative_context": alternative_context,
        "counts": counts,
        "verified_sources": verified_sources,
    }


# --- CORE ORCHESTRATION ---

def generate_hybrid_rag_news(
    user_query: str,
    api_key: str,
    language: str = "English",
    context: str | None = None,
):
    """Non-streaming hybrid RAG fact-check. Returns SUCCESS or FAIL dict."""
    try:
        _safe_print(f"\n[audit] --- AUDIT START: {user_query} ---")
        evidence = _prepare_evidence(user_query, api_key, context=context)
        payload = build_synthesis_payload(
            user_query,
            evidence["consensus_context"],
            evidence["alternative_context"],
            language,
            context=context,
        )
        raw_text = synthesize_non_stream(payload, api_key)
        return build_success_result(
            raw_text, evidence["counts"], evidence["verified_sources"]
        )
    except Exception as e:
        return handle_pipeline_error(e)


def iter_hybrid_rag_news_events(
    user_query: str,
    api_key: str,
    language: str = "English",
    context: str | None = None,
):
    """
    Streaming event generator for hybrid RAG fact-check.

    Yields:
      {"event": "phase", "phase": "planning"|"searching"|"synthesizing"|"complete"}
      {"event": "token", "text": "<incremental chunk>"}
      {"event": "result", "data": <SUCCESS or FAIL dict>}
    """
    try:
        _safe_print(f"\n[audit] --- AUDIT START (stream): {user_query} ---")

        yield {"event": "phase", "phase": "planning"}
        search_plan = generate_search_plan(user_query, api_key, context=context)
        q_factual, q_official, q_alt = unpack_search_plan(search_plan, user_query)

        yield {"event": "phase", "phase": "searching"}
        g_res, c_res, alt_res = run_parallel_searches(q_official, q_factual, q_alt)
        filtered_consensus, filtered_alternative = filter_search_results(
            user_query, g_res, c_res, alt_res
        )
        consensus_context, alternative_context, counts, verified_sources = build_source_contexts(
            filtered_consensus, filtered_alternative
        )

        yield {"event": "phase", "phase": "synthesizing"}
        payload = build_synthesis_payload(
            user_query,
            consensus_context,
            alternative_context,
            language,
            context=context,
        )

        raw_text = ""
        for kind, value in iter_synthesize_stream(payload, api_key):
            if kind == "token":
                yield {"event": "token", "text": value}
            elif kind == "done":
                raw_text = value

        result = build_success_result(raw_text, counts, verified_sources)
        yield {"event": "phase", "phase": "complete"}
        yield {"event": "result", "data": result}

    except Exception as e:
        fail = handle_pipeline_error(e)
        yield {"event": "phase", "phase": "complete"}
        yield {"event": "result", "data": fail}
