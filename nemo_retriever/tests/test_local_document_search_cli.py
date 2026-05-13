# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

from typer.testing import CliRunner

from nemo_retriever.local.__main__ import app

RUNNER = CliRunner()


def _write_manifest(index_dir, *, corpus_root, documents=None):
    index_dir.mkdir(parents=True)
    (index_dir / "lancedb").mkdir()
    manifest = {
        "schema_version": 1,
        "created_at": "2026-05-07T00:00:00+00:00",
        "updated_at": "2026-05-07T00:00:00+00:00",
        "last_query_at": None,
        "index_path": str(index_dir),
        "lancedb_uri": str(index_dir / "lancedb"),
        "lancedb_table": "local-documents",
        "input_path": str(corpus_root),
        "corpus_root": str(corpus_root),
        "include": [],
        "exclude": [],
        "max_docs": 200,
        "max_pages": None,
        "documents": documents or [],
        "documents_discovered": len(documents or []),
        "documents_processed": len(documents or []),
        "documents_skipped": [],
        "chunk_count": 1,
        "embedding_model": "nvidia/llama-nemotron-embed-1b-v2",
        "inference": {
            "requested": "local",
            "endpoint_mode": "local",
            "embed_invoke_url": None,
            "api_key_configured": False,
        },
        "warnings": [],
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_init_dry_run_discovers_supported_documents_and_skips_unsupported(tmp_path):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "guide.md").write_text("# Renewal\nThe renewal date is May 7.", encoding="utf-8")
    (corpus / "notes.tmp").write_text("ignore me", encoding="utf-8")

    result = RUNNER.invoke(app, ["init", str(corpus), "--dry-run", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["inference"]["requested"] == "local"
    assert payload["inference"]["endpoint_mode"] == "local"
    assert payload["documents_processed"] == 0
    assert [doc["relative_path"] for doc in payload["documents"]] == ["guide.md"]
    assert payload["documents_skipped"][0]["reason"] == "unsupported_format"
    assert "Dry run only" in payload["warnings"][0]


def test_init_writes_manifest_from_ingestion_summary(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("warranty limits", encoding="utf-8")
    index = tmp_path / "index"

    from nemo_retriever.local import document_search

    monkeypatch.setattr(
        document_search,
        "_run_ingestion",
        lambda *args, **kwargs: {
            "documents_processed": 1,
            "pipeline_rows": 2,
            "uploadable_chunks": 2,
            "chunk_count": 2,
            "groups": [{"input_type": "txt", "documents": 1, "rows": 2, "uploadable_chunks": 2}],
        },
    )

    result = RUNNER.invoke(app, ["init", str(corpus), "--index", str(index), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["documents_processed"] == 1
    assert payload["chunk_count"] == 2
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["documents"][0]["relative_path"] == "guide.txt"
    assert manifest["lancedb_table"] == "local-documents"


def test_init_json_writes_ingestion_progress_to_log_without_polluting_stdout(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("warranty limits", encoding="utf-8")
    index = tmp_path / ".nemo-retriever" / "local-index-test"
    progress_log = tmp_path / ".nemo-retriever" / "local-search.log"

    from nemo_retriever.local import document_search

    def fake_run_ingestion(*args, **kwargs):
        print("fake stdout progress")
        os.write(2, b"Embedding failed: fake stderr progress\n")
        return {
            "documents_processed": 1,
            "pipeline_rows": 2,
            "uploadable_chunks": 2,
            "chunk_count": 2,
            "groups": [{"input_type": "txt", "documents": 1, "rows": 2, "uploadable_chunks": 2}],
        }

    monkeypatch.setattr(document_search, "_run_ingestion", fake_run_ingestion)

    result = RUNNER.invoke(app, ["init", str(corpus), "--index", str(index), "--output", "json"])

    assert result.exit_code == 0
    assert "fake stdout progress" not in result.stdout
    assert "fake stderr progress" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["progress_log_path"] == str(progress_log.resolve())
    log_text = progress_log.read_text(encoding="utf-8")
    assert "ingest_start" in log_text
    assert "ingest_complete" in log_text
    assert "fake stdout progress" in log_text
    assert "fake stderr progress" in log_text


def test_run_ingestion_uses_graph_vdb_upload_sink(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("warranty limits", encoding="utf-8")
    index = tmp_path / "index"

    from nemo_retriever.local import document_search
    from nemo_retriever.pipeline import __main__ as pipeline_main

    discovery = document_search._discover_documents(
        corpus,
        include=[],
        exclude=[],
        max_docs=200,
        max_pages=None,
    )

    class _FakeResultFrame:
        index = [0, 1]

    class _FakeIngestor:
        def __init__(self):
            self.vdb_upload_params = []

        def vdb_upload(self, params):
            self.vdb_upload_params.append(params)
            return self

        def ingest(self):
            return "raw-result"

    fake_ingestor = _FakeIngestor()

    class _FakeExtractParams(SimpleNamespace):
        def model_copy(self, *, update=None):
            kwargs = {**self.kwargs, **(update or {})}
            return _FakeExtractParams(kwargs=kwargs)

    captured_extract_params = []

    def fake_build_extract_params(**kwargs):
        params = _FakeExtractParams(kwargs=kwargs)
        captured_extract_params.append(params)
        return params

    captured_ingestor_kwargs = {}

    monkeypatch.setattr(pipeline_main, "_build_extract_params", fake_build_extract_params)
    monkeypatch.setattr(pipeline_main, "_build_embed_params", lambda **kwargs: SimpleNamespace(kwargs=kwargs))

    def fake_build_ingestor(**kwargs):
        captured_ingestor_kwargs.update(kwargs)
        return fake_ingestor

    monkeypatch.setattr(pipeline_main, "_build_ingestor", fake_build_ingestor)
    monkeypatch.setattr(
        pipeline_main,
        "_collect_results",
        lambda run_mode, raw_result: ([], _FakeResultFrame(), 0.0, 2),
    )
    monkeypatch.setattr(
        document_search,
        "_table_info",
        lambda *args, **kwargs: {
            "readable": True,
            "uri": str(index / "lancedb"),
            "table": "local-documents",
            "table_exists": True,
            "row_count": 2,
            "error": None,
        },
    )

    summary = document_search._run_ingestion(
        discovery,
        index_path=index,
        inference_config=document_search.InferenceConfig(
            requested="remote",
            endpoint_mode="remote",
            embed_invoke_url="http://localhost:8012/v1",
            api_key_configured=False,
        ),
        embedding_model="nvidia/llama-nemotron-embed-1b-v2",
        api_key=None,
        text_chunk_max_tokens=512,
        text_chunk_overlap_tokens=32,
    )

    assert summary["documents_processed"] == 1
    assert summary["chunk_count"] == 2
    assert summary["uploadable_chunks"] == 2
    assert captured_extract_params
    assert captured_ingestor_kwargs["extract_params"].kwargs["extract_images"] is False
    assert len(fake_ingestor.vdb_upload_params) == 1
    vdb_params = fake_ingestor.vdb_upload_params[0]
    assert vdb_params.vdb_op == "lancedb"
    assert vdb_params.vdb_kwargs == {
        "uri": str(index.resolve() / "lancedb"),
        "table_name": "local-documents",
        "overwrite": True,
        "hybrid": False,
    }


def test_search_returns_agent_readable_json(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("The renewal date is May 7.", encoding="utf-8")
    stat = doc.stat()
    index = tmp_path / "index"
    _write_manifest(
        index,
        corpus_root=corpus,
        documents=[
            {
                "path": str(doc),
                "relative_path": "guide.txt",
                "input_type": "txt",
                "extension": ".txt",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "page_count": None,
            }
        ],
    )

    from nemo_retriever.local import document_search

    monkeypatch.setattr(
        document_search,
        "_table_info",
        lambda *args, **kwargs: {
            "readable": True,
            "uri": str(index / "lancedb"),
            "table": "local-documents",
            "table_exists": True,
            "row_count": 1,
            "error": None,
        },
    )

    captured_retriever_kwargs = {}

    class _FakeRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured_retriever_kwargs.update(kwargs)

        def query(self, query, top_k):
            os.write(2, b"`torch_dtype` is deprecated! Use `dtype` instead!\n")
            return [
                {
                    "text": "The renewal date is May 7.",
                    "source_id": str(doc),
                    "page_number": 1,
                    "metadata": json.dumps({"section": "Renewal"}),
                    "_distance": 0.12,
                }
            ]

    monkeypatch.setitem(sys.modules, "nemo_retriever.retriever", SimpleNamespace(Retriever=_FakeRetriever))

    result = RUNNER.invoke(
        app,
        ["search", "renewal date", "--index", str(index), "--top-k", "1", "--output", "json"],
    )

    assert result.exit_code == 0
    assert result.stdout.lstrip().startswith("{")
    assert "torch_dtype" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["results"][0]["source_file"] == str(doc)
    assert payload["results"][0]["page"] == 1
    assert payload["results"][0]["chunk_text"] == "The renewal date is May 7."
    assert "vdb" not in captured_retriever_kwargs
    assert "embedder" not in captured_retriever_kwargs
    assert "reranker" not in captured_retriever_kwargs
    assert captured_retriever_kwargs["vdb_kwargs"] == {
        "uri": str(index / "lancedb"),
        "table_name": "local-documents",
    }
    assert captured_retriever_kwargs["embed_kwargs"] == {
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "local_ingest_embed_backend": "hf",
    }
    assert captured_retriever_kwargs["rerank"] is False
    updated = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    assert updated["last_query_at"] is not None


def test_ask_reports_reused_index_when_manifest_is_fresh(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("The renewal date is May 7.", encoding="utf-8")
    stat = doc.stat()
    index = tmp_path / "index"
    _write_manifest(
        index,
        corpus_root=corpus,
        documents=[
            {
                "path": str(doc),
                "relative_path": "guide.txt",
                "input_type": "txt",
                "extension": ".txt",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "page_count": None,
            }
        ],
    )

    from nemo_retriever.local import document_search

    monkeypatch.setattr(
        document_search,
        "_table_info",
        lambda *args, **kwargs: {
            "readable": True,
            "uri": str(index / "lancedb"),
            "table": "local-documents",
            "table_exists": True,
            "row_count": 1,
            "error": None,
        },
    )
    monkeypatch.setattr(document_search, "_validate_local_inference_available", lambda: None)
    monkeypatch.setattr(
        document_search,
        "_run_ingestion_for_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reindex")),
    )

    class _FakeRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def query(self, query, top_k):
            return [{"text": "The renewal date is May 7.", "source_id": str(doc), "metadata": "{}"}]

    monkeypatch.setitem(sys.modules, "nemo_retriever.retriever", SimpleNamespace(Retriever=_FakeRetriever))

    result = RUNNER.invoke(app, ["ask", str(corpus), "renewal date", "--index", str(index), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["input_path"] == str(corpus)
    assert payload["resolved_input_path"] == str(corpus.resolve())
    assert payload["corpus_root"] == str(corpus.resolve())
    assert payload["index_action"] == "reused"
    assert payload["reused_index"] is True
    assert payload["indexed_now"] is False
    assert payload["reindexed"] is False
    assert payload["reindex_reasons"] == []


def test_ask_reuses_index_when_only_top_k_changes(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("The renewal date is May 7.", encoding="utf-8")
    stat = doc.stat()
    index = tmp_path / "index"
    _write_manifest(
        index,
        corpus_root=corpus,
        documents=[
            {
                "path": str(doc),
                "relative_path": "guide.txt",
                "input_type": "txt",
                "extension": ".txt",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "page_count": None,
            }
        ],
    )

    from nemo_retriever.local import document_search

    monkeypatch.setattr(
        document_search,
        "_table_info",
        lambda *args, **kwargs: {
            "readable": True,
            "uri": str(index / "lancedb"),
            "table": "local-documents",
            "table_exists": True,
            "row_count": 1,
            "error": None,
        },
    )
    monkeypatch.setattr(document_search, "_validate_local_inference_available", lambda: None)
    monkeypatch.setattr(
        document_search,
        "_run_ingestion_for_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reindex")),
    )

    seen_top_k = []

    class _FakeRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def query(self, query, top_k):
            seen_top_k.append(top_k)
            return [{"text": "The renewal date is May 7.", "source_id": str(doc), "metadata": "{}"}]

    monkeypatch.setitem(sys.modules, "nemo_retriever.retriever", SimpleNamespace(Retriever=_FakeRetriever))

    result = RUNNER.invoke(
        app,
        ["ask", str(corpus), "renewal date", "--index", str(index), "--top-k", "20", "--output", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["index_action"] == "reused"
    assert payload["reused_index"] is True
    assert payload["indexed_now"] is False
    assert payload["reindex_reasons"] == []
    assert seen_top_k == [20]


def test_ask_reuses_capped_index_when_only_top_k_changes(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    first = corpus / "a.txt"
    second = corpus / "b.txt"
    overflow = corpus / "c.txt"
    first.write_text("The renewal date is May 7.", encoding="utf-8")
    second.write_text("The warranty limit is 30 days.", encoding="utf-8")
    overflow.write_text("This document is intentionally outside max_docs.", encoding="utf-8")
    index = tmp_path / "index"
    _write_manifest(
        index,
        corpus_root=corpus,
        documents=[
            {
                "path": str(first),
                "relative_path": "a.txt",
                "input_type": "txt",
                "extension": ".txt",
                "size_bytes": first.stat().st_size,
                "mtime_ns": first.stat().st_mtime_ns,
                "page_count": None,
            },
            {
                "path": str(second),
                "relative_path": "b.txt",
                "input_type": "txt",
                "extension": ".txt",
                "size_bytes": second.stat().st_size,
                "mtime_ns": second.stat().st_mtime_ns,
                "page_count": None,
            },
        ],
    )
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    manifest["max_docs"] = 2
    manifest["documents_discovered"] = 3
    manifest["documents_skipped"] = [
        {
            "path": str(overflow),
            "relative_path": "c.txt",
            "reason": "max_docs_exceeded",
        }
    ]
    (index / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    from nemo_retriever.local import document_search

    monkeypatch.setattr(
        document_search,
        "_table_info",
        lambda *args, **kwargs: {
            "readable": True,
            "uri": str(index / "lancedb"),
            "table": "local-documents",
            "table_exists": True,
            "row_count": 2,
            "error": None,
        },
    )
    monkeypatch.setattr(document_search, "_validate_local_inference_available", lambda: None)
    monkeypatch.setattr(
        document_search,
        "_run_ingestion_for_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reindex")),
    )

    class _FakeRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def query(self, query, top_k):
            return [{"text": "The renewal date is May 7.", "source_id": str(first), "metadata": "{}"}]

    monkeypatch.setitem(sys.modules, "nemo_retriever.retriever", SimpleNamespace(Retriever=_FakeRetriever))

    result = RUNNER.invoke(
        app,
        [
            "ask",
            str(corpus),
            "renewal date",
            "--index",
            str(index),
            "--max-docs",
            "2",
            "--top-k",
            "20",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["index_action"] == "reused"
    assert payload["reused_index"] is True
    assert payload["indexed_now"] is False
    assert payload["reindex_reasons"] == []


def test_ask_reuses_single_file_index_when_parent_has_other_documents(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("The renewal date is May 7.", encoding="utf-8")
    sibling = corpus / "other.txt"
    sibling.write_text("Another document exists.", encoding="utf-8")
    stat = doc.stat()
    index = tmp_path / "index"
    _write_manifest(
        index,
        corpus_root=corpus,
        documents=[
            {
                "path": str(doc),
                "relative_path": "guide.txt",
                "input_type": "txt",
                "extension": ".txt",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "page_count": None,
            }
        ],
    )
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    manifest["input_path"] = str(doc)
    (index / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    from nemo_retriever.local import document_search

    monkeypatch.setattr(
        document_search,
        "_table_info",
        lambda *args, **kwargs: {
            "readable": True,
            "uri": str(index / "lancedb"),
            "table": "local-documents",
            "table_exists": True,
            "row_count": 1,
            "error": None,
        },
    )
    monkeypatch.setattr(document_search, "_validate_local_inference_available", lambda: None)
    monkeypatch.setattr(
        document_search,
        "_run_ingestion_for_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reindex")),
    )

    class _FakeRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def query(self, query, top_k):
            return [{"text": "The renewal date is May 7.", "source_id": str(doc), "metadata": "{}"}]

    monkeypatch.setitem(sys.modules, "nemo_retriever.retriever", SimpleNamespace(Retriever=_FakeRetriever))

    result = RUNNER.invoke(app, ["ask", str(doc), "renewal date", "--index", str(index), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["resolved_input_path"] == str(doc.resolve())
    assert payload["index_action"] == "reused"
    assert payload["reused_index"] is True
    assert payload["reindex_reasons"] == []


def test_ask_reports_reindex_reason_when_source_changes(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("The renewal date is May 7.", encoding="utf-8")
    index = tmp_path / "index"
    _write_manifest(
        index,
        corpus_root=corpus,
        documents=[
            {
                "path": str(doc),
                "relative_path": "guide.txt",
                "input_type": "txt",
                "extension": ".txt",
                "size_bytes": 0,
                "mtime_ns": 0,
                "page_count": None,
            }
        ],
    )

    from nemo_retriever.local import document_search

    monkeypatch.setattr(
        document_search,
        "_table_info",
        lambda *args, **kwargs: {
            "readable": True,
            "uri": str(index / "lancedb"),
            "table": "local-documents",
            "table_exists": True,
            "row_count": 1,
            "error": None,
        },
    )
    monkeypatch.setattr(document_search, "_validate_local_inference_available", lambda: None)
    ingestions = []
    monkeypatch.setattr(
        document_search,
        "_run_ingestion_for_output",
        lambda *args, **kwargs: ingestions.append(kwargs)
        or {
            "documents_processed": 1,
            "pipeline_rows": 1,
            "uploadable_chunks": 1,
            "chunk_count": 1,
            "groups": [{"input_type": "txt", "documents": 1, "rows": 1, "uploadable_chunks": 1}],
        },
    )

    class _FakeRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def query(self, query, top_k):
            return [{"text": "The renewal date is May 7.", "source_id": str(doc), "metadata": "{}"}]

    monkeypatch.setitem(sys.modules, "nemo_retriever.retriever", SimpleNamespace(Retriever=_FakeRetriever))

    result = RUNNER.invoke(app, ["ask", str(corpus), "renewal date", "--index", str(index), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["resolved_input_path"] == str(corpus.resolve())
    assert payload["index_action"] == "reindexed"
    assert payload["reused_index"] is False
    assert payload["indexed_now"] is True
    assert payload["reindexed"] is True
    assert payload["reindex_reasons"] == ["documents_changed"]
    assert len(ingestions) == 1


def test_status_reports_health_and_staleness(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("content", encoding="utf-8")
    stat = doc.stat()
    index = tmp_path / "index"
    _write_manifest(
        index,
        corpus_root=corpus,
        documents=[
            {
                "path": str(doc),
                "relative_path": "guide.txt",
                "input_type": "txt",
                "extension": ".txt",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "page_count": None,
            }
        ],
    )

    from nemo_retriever.local import document_search

    monkeypatch.setattr(
        document_search,
        "_table_info",
        lambda *args, **kwargs: {
            "readable": True,
            "uri": str(index / "lancedb"),
            "table": "local-documents",
            "table_exists": True,
            "row_count": 3,
            "error": None,
        },
    )

    result = RUNNER.invoke(app, ["status", "--index", str(index), "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["chunks"] == 3
    assert payload["documents"]["processed"] == 1
    assert payload["health_checks"][0]["name"] == "manifest_present"
