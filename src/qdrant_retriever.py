import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from src.embedder import Embedder
from src.schema import DocumentChunk, SearchResult
from tqdm.auto import tqdm


class QdrantRetriever:
    """Vector database retrieval using Qdrant & ONNX Embedder (supports Docker or in-memory mode)."""

    def __init__(
        self,
        collection_name: str = "tw3k_transcripts",
        url: str = "http://localhost:6333",
        prefer_grpc: bool = False,
        in_memory: bool = False,
        embedder: Optional[Embedder] = None,
        model_name: str = "Xenova/all-MiniLM-L6-v2",
    ):
        self.collection_name = collection_name
        self.embedder = embedder or Embedder(model_name=model_name)
        self.vector_dim = self.embedder.get_embedding_dimension()

        # Connect to Qdrant server or fallback to in-memory mode
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            try:
                self.client = QdrantClient(url=url, prefer_grpc=prefer_grpc, timeout=5.0)
                # Test connection by fetching collections
                self.client.get_collections()
            except Exception:
                # Fallback to in-memory mode if Docker container is not active
                self.client = QdrantClient(":memory:")

        # Create Qdrant collection if it does not exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64):
        """Encode document chunks using ONNX Embedder and upsert vectors + metadata into Qdrant."""
        if not chunks:
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

        # Execute vector similarity query in Qdrant using query_points (qdrant-client >= 1.9)
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
