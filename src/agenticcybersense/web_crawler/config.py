"""Webcrawler configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Use the web_crawler directory regardless of the process working directory.
_BASE_DIR = Path(__file__).parent

# Path to the Excel file containing site configurations.
SITES_FILE: str = os.getenv(
    "CRAWLER_SITES_FILE",
    str(_BASE_DIR / "config" / "sites.xlsx"),
)

# Blacklisted URLs that must not be crawled.
BLACKLIST: list[str] = []

# Depth limits per crawl level.
# None means unlimited.
DEPTH_LIMITS: dict[int, int | None] = {
    0: None,  # Depth 0: unlimited
    1: 10,  # Depth 1: randomly crawl up to 10 links
    2: 0,  # Depth 2: disabled
}

# Concurrent site count.
# - Low RAM / weak CPU -> 2
# - Mid-range machine  -> 3 (recommended)
# - Strong server      -> 5
CONCURRENT_SITES = 3

# Timeout for inactive crawls in seconds.
INACTIVITY_TIMEOUT = 180

# Output paths are always created under web_crawler/output.
OUTPUT_FILE = str(_BASE_DIR / "output" / "latest_results.json")
HISTORY_FILE = str(_BASE_DIR / "output" / "crawl_history.db")

# Incremental crawling.
ENABLE_INCREMENTAL = True

# True  = ignore hash checks and always crawl everything
# False = skip unchanged sites
FORCE_FULL_CRAWL = False

# Scheduler
SCHEDULE_HOUR: int = int(os.getenv("CRAWLER_SCHEDULE_HOUR", "0"))
SCHEDULE_MINUTE: int = int(os.getenv("CRAWLER_SCHEDULE_MINUTE", "0"))
