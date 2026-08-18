from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import download_model as download_script
from scripts import ingest_qdrant
from tw3k_ai_assistant.database import initialization as db_init


qdrant_module = importlib.import_module("tw3k_ai_assistant.retrieval.qdrant")
embedder_module = importlib.import_module("tw3k_ai_assistant.retrieval.embedder")


class FakeEmbedder:
    def get_embedding_dimension(self) -> int:
        return 3


class RecordingQdrantClient:
    instances: list["RecordingQdrantClient"] = []

    def __init__(self, location=None, url=None, **kwargs):
        self.location = location
        self.url = url
        self.kwargs = kwargs
        self.collections: set[str] = set()
        self.instances.append(self)

    def get_collections(self):
        return SimpleNamespace(collections=[])

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections.add(collection_name)

    def get_collection(self, collection_name):
        return SimpleNamespace(points_count=0)


@pytest.mark.parametrize(
    ("environment", "expected_url", "expected_collection"),
    (
        ({}, "http://localhost:6333", "tw3k_transcripts"),
        (
            {
                "QDRANT_URL": "http://qdrant.internal:6333",
                "QDRANT_COLLECTION": "custom_transcripts",
            },
            "http://qdrant.internal:6333",
            "custom_transcripts",
        ),
    ),
)
def test_qdrant_environment_reaches_configured_client(
    monkeypatch, environment, expected_url, expected_collection
):
    for name in ("QDRANT_URL", "QDRANT_COLLECTION"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    RecordingQdrantClient.instances = []
    monkeypatch.setattr(qdrant_module, "QdrantClient", RecordingQdrantClient)

    retriever = qdrant_module.QdrantRetriever(embedder=FakeEmbedder())

    assert retriever.url == expected_url
    assert retriever.collection_name == expected_collection
    assert RecordingQdrantClient.instances[0].url == expected_url
    assert expected_collection in RecordingQdrantClient.instances[0].collections


def test_unavailable_qdrant_does_not_fall_back_to_memory(monkeypatch):
    calls = []

    class FailingQdrantClient:
        def __init__(self, location=None, url=None, **kwargs):
            calls.append((location, url))

        def get_collections(self):
            raise OSError("connection refused")

        def close(self):
            pass

    monkeypatch.setattr(qdrant_module, "QdrantClient", FailingQdrantClient)

    with pytest.raises(ConnectionError, match="configured Qdrant server") as error:
        qdrant_module.QdrantRetriever(
            embedder=FakeEmbedder(),
            url="http://unavailable.example:6333",
        )

    assert "unavailable.example:6333" in str(error.value)
    assert calls == [(None, "http://unavailable.example:6333")]


class FakeIngestionRetriever:
    initial_count = 0
    instances: list["FakeIngestionRetriever"] = []

    def __init__(self, *, collection_name, url):
        self.collection_name = collection_name
        self.url = url
        self.count = type(self).initial_count
        self.index_calls = []
        self.instances.append(self)

    def get_point_count(self):
        return self.count

    def index_chunks(self, chunks, *, batch_size, force):
        self.index_calls.append((len(chunks), batch_size, force))
        self.count = len(chunks)


def test_ingestion_populates_empty_collection_with_configured_batch(monkeypatch):
    FakeIngestionRetriever.initial_count = 0
    FakeIngestionRetriever.instances = []
    monkeypatch.setattr(ingest_qdrant, "QdrantRetriever", FakeIngestionRetriever)
    monkeypatch.setattr(ingest_qdrant, "load_dataset", lambda path: ["one", "two"])

    status = ingest_qdrant.run(
        environ={
            "QDRANT_URL": "http://configured-qdrant:6333",
            "QDRANT_COLLECTION": "configured_collection",
            "QDRANT_BATCH_SIZE": "7",
        }
    )

    assert status == 0
    retriever = FakeIngestionRetriever.instances[0]
    assert (retriever.url, retriever.collection_name) == (
        "http://configured-qdrant:6333",
        "configured_collection",
    )
    assert retriever.index_calls == [(2, 7, False)]


def test_ingestion_defaults_to_skip_and_force_is_explicit(monkeypatch):
    monkeypatch.setattr(ingest_qdrant, "QdrantRetriever", FakeIngestionRetriever)
    monkeypatch.setattr(ingest_qdrant, "load_dataset", lambda path: ["one", "two"])

    FakeIngestionRetriever.initial_count = 4
    FakeIngestionRetriever.instances = []
    ingest_qdrant.run(environ={})
    assert FakeIngestionRetriever.instances[0].index_calls == []

    FakeIngestionRetriever.initial_count = 4
    FakeIngestionRetriever.instances = []
    ingest_qdrant.run(environ={"QDRANT_FORCE_REINDEX": "true"})
    assert FakeIngestionRetriever.instances[0].index_calls == [(2, 64, True)]


def test_ingestion_rejects_invalid_settings_before_connecting(monkeypatch):
    class UnexpectedRetriever:
        def __init__(self, **kwargs):
            raise AssertionError("Qdrant was contacted before config validation")

    monkeypatch.setattr(ingest_qdrant, "QdrantRetriever", UnexpectedRetriever)
    monkeypatch.setattr(ingest_qdrant, "load_dataset", lambda path: ["one"])

    with pytest.raises(ValueError, match="positive integer"):
        ingest_qdrant.run(environ={"QDRANT_BATCH_SIZE": "0"})
    with pytest.raises(ValueError, match="QDRANT_FORCE_REINDEX"):
        ingest_qdrant.run(environ={"QDRANT_FORCE_REINDEX": "maybe"})


def test_ingestion_rejects_empty_dataset_before_connecting(monkeypatch):
    class UnexpectedRetriever:
        def __init__(self, **kwargs):
            raise AssertionError("Qdrant was contacted for an empty dataset")

    monkeypatch.setattr(ingest_qdrant, "QdrantRetriever", UnexpectedRetriever)
    monkeypatch.setattr(ingest_qdrant, "load_dataset", lambda path: [])

    with pytest.raises(RuntimeError, match="no indexable chunks"):
        ingest_qdrant.run(environ={})


class FakeDatabase:
    def __init__(self):
        self.tables: set[str] = set()
        self.feedback_columns = {"id", "conversation_id", "score", "timestamp"}
        self.conversation_rows = ["existing conversation"]
        self.feedback_rows = ["existing feedback"]
        self.statements: list[str] = []
        self.commit_count = 0


class FakeCursor:
    def __init__(self, database: FakeDatabase):
        self.database = database
        self.results = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, statement, params=None):
        normalized = " ".join(statement.split()).lower()
        self.database.statements.append(normalized)
        if normalized.startswith("create table if not exists conversations"):
            self.database.tables.add("conversations")
        elif normalized.startswith("create table if not exists feedback"):
            self.database.tables.add("feedback")
        elif "from information_schema.columns" in normalized:
            self.results = [(column,) for column in self.database.feedback_columns]
        elif normalized.startswith("drop table"):
            raise AssertionError("normal schema initialization attempted a drop")

    def fetchall(self):
        return self.results


class FakeConnection:
    def __init__(self, database: FakeDatabase):
        self.database = database

    def cursor(self):
        return FakeCursor(self.database)

    def commit(self):
        self.database.commit_count += 1

    def close(self):
        pass


def test_database_initialization_is_repeatable_without_dropping_rows(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(db_init, "get_db_connection", lambda: FakeConnection(database))

    db_init.init_db()
    db_init.init_feedback()
    db_init.init_db()
    db_init.init_feedback()

    assert database.tables == {"conversations", "feedback"}
    assert database.conversation_rows == ["existing conversation"]
    assert database.feedback_rows == ["existing feedback"]
    assert database.commit_count == 4
    assert not any(statement.startswith("drop table") for statement in database.statements)


def test_complete_model_cache_skips_hugging_face(tmp_path, monkeypatch):
    model_dir = tmp_path / "Xenova" / "all-MiniLM-L6-v2"
    model_dir.mkdir(parents=True)
    (model_dir / "tokenizer.json").write_text("cached tokenizer", encoding="utf-8")
    (model_dir / "model.onnx").write_bytes(b"cached model")

    def fail_if_called(**kwargs):
        raise AssertionError("complete cache contacted Hugging Face")

    monkeypatch.setattr(download_script, "list_repo_files", fail_if_called)
    monkeypatch.setattr(download_script, "hf_hub_download", fail_if_called)

    assert download_script.download(dest=tmp_path) == model_dir


def test_partial_model_cache_downloads_missing_files_without_network(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_files = {
        "tokenizer.json": remote_root / "tokenizer.json",
        "onnx/model.onnx": remote_root / "model.onnx",
        "onnx/model.onnx_data": remote_root / "model.onnx_data",
    }
    for source in remote_files.values():
        source.write_bytes(b"model file")

    existing = cache_root / "example" / "model" / "tokenizer.json"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already present")
    calls = []

    monkeypatch.setattr(download_script, "list_repo_files", lambda **kwargs: list(remote_files))

    def fake_hf_download(*, repo_id, filename):
        calls.append(filename)
        return str(remote_files[filename])

    monkeypatch.setattr(download_script, "hf_hub_download", fake_hf_download)

    result = download_script.download(repo="example/model", dest=cache_root)

    assert (result / "tokenizer.json").read_bytes() == b"already present"
    assert (result / "model.onnx").read_bytes() == b"model file"
    assert (result / "model.onnx_data").read_bytes() == b"model file"
    assert calls == ["onnx/model.onnx", "onnx/model.onnx_data"]


def test_embedder_reads_model_name_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_MODEL", "example/custom-model")
    monkeypatch.setattr(embedder_module.Embedder, "_ensure_model_downloaded", lambda self: None)
    monkeypatch.setattr(embedder_module.Tokenizer, "from_file", lambda path: object())

    class FakeSession:
        def get_inputs(self):
            return []

    monkeypatch.setattr(embedder_module.ort, "InferenceSession", lambda *args, **kwargs: FakeSession())

    embedder = embedder_module.Embedder(models_dir=tmp_path)

    assert embedder.model_name == "example/custom-model"
    assert embedder.dest_dir == tmp_path / "example" / "custom-model"
