from __future__ import annotations

import os
import uuid

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_QDRANT_INTEGRATION") != "1",
        reason="set RUN_QDRANT_INTEGRATION=1 to use a real Qdrant service",
    ),
]


def test_real_qdrant_create_upsert_update_and_search() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    client = QdrantClient(
        url=os.environ.get("QDRANT_INTEGRATION_URL", "http://localhost:6333"),
        timeout=10,
    )
    collection = f"tw3k_integration_{uuid.uuid4().hex}"
    first_id = str(uuid.uuid4())
    second_id = str(uuid.uuid4())
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        )
        client.upsert(
            collection_name=collection,
            wait=True,
            points=[models.PointStruct(id=first_id, vector=[1, 0, 0, 0], payload={"version": 1})],
        )
        assert client.count(collection, exact=True).count == 1

        client.upsert(
            collection_name=collection,
            wait=True,
            points=[
                models.PointStruct(id=first_id, vector=[0, 1, 0, 0], payload={"version": 2}),
                models.PointStruct(id=second_id, vector=[1, 0, 0, 0], payload={"version": 1}),
            ],
        )
        assert client.count(collection, exact=True).count == 2
        updated = client.retrieve(collection, ids=[first_id], with_payload=True, with_vectors=True)[0]
        assert updated.payload == {"version": 2}
        assert updated.vector == [0.0, 1.0, 0.0, 0.0]

        result = client.query_points(
            collection_name=collection,
            query=[1, 0, 0, 0],
            limit=1,
            with_payload=True,
        )
        assert str(result.points[0].id) == second_id
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
