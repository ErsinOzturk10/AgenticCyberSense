"""Scrapy items for the cyber threat intelligence crawler."""

from __future__ import annotations

import scrapy


class CrawlPageItem(scrapy.Item):
    """Represents a single crawled page with extracted content."""

    url = scrapy.Field()
    title = scrapy.Field()
    html = scrapy.Field()
    main_content = scrapy.Field()
    links = scrapy.Field()
    metadata = scrapy.Field()
    structured_data = scrapy.Field()
    extraction_type = scrapy.Field()
    site_url = scrapy.Field()
    depth = scrapy.Field()
    method = scrapy.Field()
