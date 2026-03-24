from urllib.parse import urlparse

SOURCE_METADATA = {
    # Government & Authorities
    "pib.gov.in": {"name": "PIB Fact Check", "trust_score": 99, "category": "Govt of India"},
    "india.gov.in": {"name": "National Portal", "trust_score": 99, "category": "Govt of India"},
    "mp.gov.in": {"name": "MP State Portal", "trust_score": 98, "category": "State Government"},
    "rbi.org.in": {"name": "Reserve Bank of India", "trust_score": 99, "category": "Central Bank"},
    "eci.gov.in": {"name": "Election Commission", "trust_score": 99, "category": "Constitutional Body"},
    "mohfw.gov.in": {"name": "Health Ministry", "trust_score": 98, "category": "Govt of India"},
    "who.int": {"name": "World Health Org", "trust_score": 98, "category": "Global Authority"},
    
    # State Broadcasters & Wires
    "ddnews.gov.in": {"name": "DD News", "trust_score": 96, "category": "Public Broadcaster"},
    "ptinews.com": {"name": "Press Trust of India", "trust_score": 96, "category": "National Wire"},
    "aniin.com": {"name": "ANI", "trust_score": 90, "category": "News Wire"},
    
    # Fact Checkers & Media
    "boomlive.in": {"name": "BOOM Live", "trust_score": 95, "category": "IFCN Certified"},
    "reuters.com": {"name": "Reuters", "trust_score": 97, "category": "International Agency"},
    "thehindu.com": {"name": "The Hindu", "trust_score": 94, "category": "Newspaper of Record"},
    "indianexpress.com": {"name": "Indian Express", "trust_score": 93, "category": "Newspaper of Record"},
    "factly.in": {"name": "Factly", "trust_score": 94, "category": "IFCN Certified"},
    "altnews.in": {"name": "Alt News", "trust_score": 94, "category": "Independent Fact-Check"},
}

def get_source_profile(url: str):
    """Extracts domain and returns robust reputation data."""
    try:
        # Extract the domain, handling potential subdomains cleanly
        domain = urlparse(url).netloc.lower().replace("www.", "")
        
        # Check if any of our trusted root domains are inside the extracted domain
        for key, data in SOURCE_METADATA.items():
            if key in domain:
                return data
                
        # Fallback for unknown sources (MUST match the keys of the known sources!)
        return {
            "name": domain if domain else "External Source",
            "trust_score": 50,
            "category": "General Web"
        }
    except Exception:
        # Failsafe in case a completely broken URL is passed
        return {
            "name": "External Source",
            "trust_score": 50,
            "category": "General Web"
        }