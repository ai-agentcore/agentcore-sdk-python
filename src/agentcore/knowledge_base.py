"""Retrieval from an existing native AgentCore knowledge base."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from anyio.from_thread import BlockingPortal

from agentcore.controlplane.client import AgentCoreControlPlane
from agentcore.errors import InvocationError


@dataclass(frozen=True)
class KnowledgeBaseChunk:
    content: str
    asset_id: str | None = None
    asset_name: str | None = None
    file_name: str | None = None
    chunk_seq: int | None = None
    chunk_token: str | None = None
    score: float | None = None
    scores: Mapping[str, float] = field(default_factory=dict)
    title_path: str | None = None
    source_location: str | None = None


@dataclass(frozen=True)
class KnowledgeBaseSearchResult:
    chunks: tuple[KnowledgeBaseChunk, ...]
    time_spent: int | None = None
    score_type: str | None = None


class AsyncKnowledgeBase:
    def __init__(
        self,
        knowledge_base_id: str,
        provider: Callable[[], Awaitable[AgentCoreControlPlane]],
    ) -> None:
        if not knowledge_base_id.strip():
            raise ValueError("knowledge_base_id must not be blank")
        self.knowledge_base_id = knowledge_base_id
        self._provider = provider

    async def search(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        metadata_filter: str | None = None,
        rerank: bool | None = None,
        rrf_k: int | None = None,
        rank_algorithm: str | None = None,
        vector_weight: float | None = None,
        rerank_model: str | None = None,
    ) -> KnowledgeBaseSearchResult:
        """Return ranked chunks. Omitted options inherit the knowledge base configuration."""
        if not query.strip():
            raise ValueError("query must not be blank")
        body = {
            key: value
            for key, value in {
                "query": query,
                "mode": mode,
                "topK": top_k,
                "metadataFilter": metadata_filter,
                "rerank": rerank,
                "rrfK": rrf_k,
                "rankAlgorithm": rank_algorithm,
                "vectorWeight": vector_weight,
                "rerankModel": rerank_model,
            }.items()
            if value is not None
        }
        provider = await self._provider()
        data = await provider.search_knowledge_base(self.knowledge_base_id, body)
        chunks = data.get("chunks")
        if not isinstance(chunks, list):
            raise InvocationError("SearchKnowledgeBase returned invalid data.chunks")
        result = []
        for chunk in chunks:
            if not isinstance(chunk, Mapping) or not isinstance(chunk.get("content"), str):
                raise InvocationError("SearchKnowledgeBase returned a chunk without content")
            result.append(
                KnowledgeBaseChunk(
                    content=chunk["content"],
                    asset_id=chunk.get("assetId"),
                    asset_name=chunk.get("assetName"),
                    file_name=chunk.get("fileName"),
                    chunk_seq=chunk.get("chunkSeq"),
                    chunk_token=chunk.get("chunkToken"),
                    score=chunk.get("score"),
                    scores=chunk.get("scores") or {},
                    title_path=chunk.get("titlePath"),
                    source_location=chunk.get("sourceLocation"),
                )
            )
        return KnowledgeBaseSearchResult(
            tuple(result), data.get("timeSpent"), data.get("scoreType")
        )


class KnowledgeBase:
    """Synchronous facade; use core.knowledge_base() to bind it to a Core."""

    def __init__(
        self,
        client: AsyncKnowledgeBase,
        portal: BlockingPortal,
        operation: Callable[[], AbstractContextManager[Any]],
    ) -> None:
        self._client, self._portal, self._operation = client, portal, operation

    @property
    def knowledge_base_id(self) -> str:
        return self._client.knowledge_base_id

    def search(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        metadata_filter: str | None = None,
        rerank: bool | None = None,
        rrf_k: int | None = None,
        rank_algorithm: str | None = None,
        vector_weight: float | None = None,
        rerank_model: str | None = None,
    ) -> KnowledgeBaseSearchResult:
        with self._operation():
            return self._portal.call(
                partial(
                    self._client.search,
                    query,
                    mode=mode,
                    top_k=top_k,
                    metadata_filter=metadata_filter,
                    rerank=rerank,
                    rrf_k=rrf_k,
                    rank_algorithm=rank_algorithm,
                    vector_weight=vector_weight,
                    rerank_model=rerank_model,
                )
            )
