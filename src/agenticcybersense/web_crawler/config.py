"""Webcrawler configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Use the web_crawler directory regardless of the process working directory.
_BASE_DIR = Path(__file__).parent


def _read_int_env(name: str, default: int) -> int:
    """Parse integer env vars while tolerating common markdown copy/paste artifacts."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.strip().strip("`")
    if normalized_value == "":
        return default

    try:
        return int(normalized_value)
    except ValueError as exc:
        msg = f"Invalid integer value for {name}: {raw_value!r}"
        raise ValueError(msg) from exc

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
SCHEDULE_HOUR: int = _read_int_env("CRAWLER_SCHEDULE_HOUR", 0)
SCHEDULE_MINUTE: int = _read_int_env("CRAWLER_SCHEDULE_MINUTE", 0)
