"""Schemas for security findings and vulnerabilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Severity(str, Enum):
    """Severity levels for findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SourceType(str, Enum):
    """Types of intelligence sources."""

    DOCUMENTATION = "documentation"
    WEBSITE = "website"
    TELEGRAM = "telegram"
    FORUM = "forum"
    DARK_WEB = "dark_web"
    OTHER = "other"


@dataclass
class SourceRef:
    """Reference to the source of a finding."""

    source_type: SourceType
    url: str | None = None
    name: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_content: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Finding:
    """A security finding or threat intelligence item."""

    title: str
    description: str
    severity: Severity
    source: SourceRef
    finding_id: str = field(default_factory=lambda: "")
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)  # IOCs
    recommendations: list[str] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)
    verified: bool = False

    def __post_init__(self) -> None:
        """Generate finding ID if not provided."""
        if not self.finding_id:
            content = f"{self.title}{self.description}{self.created_at.isoformat()}"
            self.finding_id = hashlib.sha256(content.encode()).hexdigest()[:12]
