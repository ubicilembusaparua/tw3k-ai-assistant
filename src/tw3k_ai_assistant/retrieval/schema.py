from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DocumentChunk:
    """Represents a text chunk stored in the retrieval index."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a retrieved result with relevance score and rank."""
    chunk: DocumentChunk
    score: float
    rank: int
