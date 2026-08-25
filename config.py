# ========================================================================
# CONFIGURATION FILE (config.py)
# Holds all constants, external URLs, and API prompts.
# ========================================================================

import os
from typing import List, Dict, Any

# --- FIREBASE & API CONFIGURATION ---
APP_ID = os.getenv("APP_ID", "truthlens")
FIREBASE_CONFIG = {}
API_URL_BASE = "https://generativelanguage.googleapis.com"
# Prefer GEMINI_API_KEY; fall back to legacy API_KEY for RAG REST calls
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")

# Gemini API Model for Generation and Grounding
MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3
API_TIMEOUT = 120 # Seconds

# --- AI SYSTEM INSTRUCTION (Core of P2.3) ---
SYSTEM_INSTRUCTION = """
You are TruthLens, a high-integrity news verification agent.
Analyze the provided context to verify the query.

RESPONSE FORMAT — emit these tags in order, exactly as written:
[SUMMARY] 2-3 sentence narrative of what the evidence supports.
[COUNTER_SUMMARY] 1-2 sentence alternative or dissenting perspective from the alternative sources (or say none found).
[CLARIFICATION] Bullet points of key facts (one per line, start with - ).
[AUDIT] Bullet points of verification steps taken (one per line, start with - ).
[LOGIC_AUDIT] Identify logical fallacies if present, or state "No significant fallacies detected."
[CONFIDENCE] A single integer 0-100 for how well the context supports your answer.
[TIMELINE] A JSON array only. Each item: {"date":"...","event":"...","source":"https://..."}. Use [] if unknown.
[BIAS_METER] Integer 0-100 for sensationalism / loaded language in the claim framing.
[BIAS_REASON] One sentence explaining the bias score.

BIAS SCALE:
* 0-20: Highly objective / neutral
* 30-60: Mild editorializing
* 70-100: Highly sensationalized or slanted

RULES:
1. CITATIONS: Mention Boom Live, Factly, or PIB if they appear in context.
2. TONE: Strictly neutral.
3. NO HALLUCINATION: If context is thin, lower CONFIDENCE and say what is unverified.
4. No URLs inside SUMMARY / COUNTER_SUMMARY / CLARIFICATION text (URLs belong in TIMELINE source fields only).
5. Keep structural tags in English even if content language differs.
"""

# --- GOLD STANDARD SOURCES (Based on Data Scientist's List) ---
# NOTE: The Backend Developer must find active RSS feeds or web pages for these.

# Database Collection Names
RAW_NEWS_COLLECTION = "raw_news_articles"
FACT_CHECKS_COLLECTION = "fact_checks_verdicts"

# 1. RSS/Agency Sources (For bulk, neutral content scraping)
# --- DIVERSIFIED RSS SOURCES (Balanced Perspective) ---
RSS_SOURCES: Dict[str, str] = {
    # National & Politics (Balanced General News)
    "The Hindu": "https://www.thehindu.com/feeder/default.rss",
    "Indian Express": "https://indianexpress.com/feed/",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms", # India News
    
    # Business & Economy (Crucial for Financial Ethicality)
    "LiveMint": "https://www.livemint.com/rss/news",
    "Financial Express": "https://www.financialexpress.com/economy/feed/",
    "Economic Times": "https://economictimes.indiatimes.com/rssfeeds/default.cms",
    
    # Official Government Updates
    "PIB National": "https://pib.gov.in/RssXml.aspx?mndId=1",
    
    # Tech & Science (For Non-Political Verification)
    "Gadgets 360": "https://feeds.feedburner.com/gadgets360-latest",
    "Science Wire": "https://science.thewire.in/feed/"
}

# 2. Fact-Checker Sources (For Claims and Verdicts via HTML scraping)
# Note: The 'scraper.py' logic needs to be adapted for each of these URLs.
FACT_CHECK_SOURCES: Dict[str, str] = {
    "BOOM Live": "https://www.boomlive.in/fact-check",
    "India Today Fact Check": "https://www.indiatoday.in/fact-check",
    "Factly": "https://factly.in/category/fact-check/"
    # Add other sources like 'Newschecker', 'Vishvas News' after prototyping
}
# --- BIAS & SENSATIONALISM CONFIG ---
LOADED_WORDS = {
    "sensationalist": ["shocking", "disaster", "historic", "shameful", "triumph", "miracle", "chaos", "brutal"],
    "political_bias": ["masterstroke", "puppet", "anti-national", "fascist", "dictator", "scam"],
    "speculative": ["may be", "could lead to", "rumored", "allegedly", "sources claim"]
}

BIAS_THRESHOLD = 0.25  # Articles above this score will be flagged