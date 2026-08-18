import os
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm.auto import tqdm

from src.embedder import Embedder
from src.schema import DocumentChunk, SearchResult


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "tw3k_transcripts"
DEFAULT_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2"


class QdrantRetriever:
    """Vector retrieval using a configured Qdrant server or explicit memory mode.

    The external-server path is intentionally fail-fast. An unavailable
    configured server must never look like an empty in-memory collection.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
        prefer_grpc: bool = False,
        in_memory: bool = False,
        embedder: Optional[Embedder] = None,
        model_name: Optional[str] = None,
    ):
        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION")
            or DEFAULT_QDRANT_COLLECTION
        )
        self.url = url or os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
        configured_model = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self.embedder = embedder or Embedder(model_name=configured_model)
        self.vector_dim = self.embedder.get_embedding_dimension()
        self.in_memory = in_memory

        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            try:
                self.client = QdrantClient(
                    url=self.url,
                    prefer_grpc=prefer_grpc,
                    timeout=5.0,
                )
                self.client.get_collections()
            except Exception as exc:
                close = getattr(getattr(self, "client", None), "close", None)
                if close is not None:
                    close()
                raise ConnectionError(
                    f"Unable to connect to configured Qdrant server at {self.url!r}. "
                    "Check QDRANT_URL and confirm the service is reachable."
                ) from exc

        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as exc:
            if in_memory:
                raise
            raise ConnectionError(
                f"Unable to initialize Qdrant collection {self.collection_name!r} "
                f"on {self.url!r}."
            ) from exc

    def get_point_count(self) -> int:
        """Returns total vector points stored in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as exc:
            if not self.in_memory:
                raise ConnectionError(
                    f"Unable to read Qdrant collection {self.collection_name!r} "
                    f"from {self.url!r}."
                ) from exc
            raise

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64, force: bool = False):
        """Encode document chunks using ONNX Embedder and upsert vectors + metadata into Qdrant."""
        if not chunks:
            return

        # If force is True, recreate collection to purge stale vectors from previous dataset versions
        if force:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
        elif self.get_point_count() > 0:
            print(f"Skipping indexing: Collection '{self.collection_name}' already contains {self.get_point_count()} points.")
            return

        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embedder.encode_batch(texts, normalize=True)

            points = []
            for idx, (chunk, emb) in enumerate(zip(batch, embeddings)):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
                payload = {
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    **chunk.metadata,
                }
                points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))

            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search top-k most relevant chunks in Qdrant vector database using ONNX Embedder."""
        if not query.strip():
            return []

        query_vector = self.embedder.encode(query, normalize=True).tolist()

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            )
            hits = response.points
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )

        results = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id", str(hit.id))
            content = payload.get("content", "")
            metadata = {k: v for k, v in payload.items() if k not in ("chunk_id", "content")}
            
            chunk = DocumentChunk(id=chunk_id, content=content, metadata=metadata)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(hit.score),
                    rank=rank,
                )
            )
        return results
