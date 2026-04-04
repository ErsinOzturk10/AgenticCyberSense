"""
Configuration — v2
"""

# BLACKLIST
BLACKLIST = [
   
]

# DEPTH LIMITS
DEPTH_LIMITS = {
    0: None,   # Depth 0: SINIRSIZ
    1: 10,     # Depth 1: Random 10 link
    2: 0,      # Depth 2: Yok
}

# ── YENİ: Eş zamanlı site sayısı ─────────────────────────────────── #
#  Kaç site paralel işlensin?
#  - Düşük RAM / zayıf CPU → 2
#  - Orta donanım          → 3  (önerilen)
#  - Güçlü GPU sunucu      → 5
#  Not: Ollama tek thread çalışır; asıl kazanım Playwright/network tarafında.
CONCURRENT_SITES = 3

INACTIVITY_TIMEOUT = 180
# OUTPUT
OUTPUT_FILE  = "output/latest_results.json"
HISTORY_FILE = "crawl_history.db"   # artık SQLite, .json değil

# INCREMENTAL CRAWLING
ENABLE_INCREMENTAL = True

# FORCE FULL CRAWL
# True  = Hash kontrolünü atla, her zaman full crawl
# False = Değişmemiş siteleri atla (önerilen)
FORCE_FULL_CRAWL = False

# RAG AYARLARI
CHROMA_DB_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"  # Ollama'da çalışır
CHUNK_SIZE = 500        # kelime
CHUNK_OVERLAP = 50      # kelime
ENABLE_RAG = True       # False yapılırsa RAG indexleme atlanır
RAG_TOP_K = 5           # Kaç sonuç dönsün