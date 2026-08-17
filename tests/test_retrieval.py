"""Retrieval: BM25, dense search, rank fusion, MMR and metadata filtering."""

from __future__ import annotations

import numpy as np
import pytest

from sanad.config import RetrievalSettings
from sanad.documents.models import Chunk
from sanad.errors import RetrievalError
from sanad.rag.retriever import (
    HybridRetriever,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from sanad.rag.store import LexicalIndex, SearchIndex
from tests.conftest import bag_of_words

CORPUS = [
    ("D1", "contract.pdf", "Section 3", "Either party may terminate this contract on sixty days notice."),
    ("D1", "contract.pdf", "Section 2", "The employer shall pay a monthly salary of SAR 18,500."),
    ("D1", "contract.pdf", "Section 4", "This agreement is governed by the law of Riyadh jurisdiction."),
    ("D2", "supply.pdf", "Clause 9", "The supplier shall deliver the goods within thirty days."),
    ("D2", "supply.pdf", "Clause 4", "A penalty applies for late delivery of the goods."),
]


def make_chunks():
    return [
        Chunk(
            chunk_id=f"{doc}-C{index:04d}",
            doc_id=doc,
            filename=name,
            index=index,
            text=text,
            page_start=index + 1,
            page_end=index + 1,
            section=section,
        )
        for index, (doc, name, section, text) in enumerate(CORPUS)
    ]


@pytest.fixture
def index() -> SearchIndex:
    chunks = make_chunks()
    built = SearchIndex()
    built.build(chunks, [bag_of_words(chunk.text) for chunk in chunks])
    return built


class StubEmbedder:
    def embed_query(self, text: str):
        return bag_of_words(text)


class TestLexicalIndex:
    def test_ranks_the_literal_match_first(self) -> None:
        lexical = LexicalIndex()
        lexical.build(make_chunks())

        results = lexical.search("termination notice", limit=3)
        assert results
        assert results[0][0] == 0

    def test_matches_the_section_label(self) -> None:
        lexical = LexicalIndex()
        lexical.build(make_chunks())

        # "Clause 9" appears only in the section label, never in the body text.
        results = lexical.search("Clause 9", limit=3)
        assert results and results[0][0] == 3

    def test_normalises_arabic_digits_and_spelling(self) -> None:
        chunks = [
            Chunk("D1-C0000", "D1", "عقد.pdf", 0, "يستحق الموظف أجراً قدره 18500 ريال", 1, 1, "المادة ٢")
        ]
        lexical = LexicalIndex()
        lexical.build(chunks)

        # Query written with Arabic-Indic digits still reaches the ASCII text.
        assert lexical.search("المادة ٢", limit=1)
        assert lexical.search("المادة 2", limit=1)

    def test_empty_index_returns_nothing(self) -> None:
        assert LexicalIndex().search("anything", limit=5) == []


class TestSearchIndex:
    def test_rejects_a_vector_count_mismatch(self) -> None:
        chunks = make_chunks()
        with pytest.raises(ValueError, match="one vector per chunk"):
            SearchIndex().build(chunks, [bag_of_words(chunks[0].text)])

    def test_dense_search_orders_by_cosine_similarity(self, index: SearchIndex) -> None:
        results = index.dense_search(bag_of_words("terminate the contract"), limit=3)

        assert results[0][0] == 0
        scores = [score for _position, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_rejects_a_query_of_the_wrong_dimension(self, index: SearchIndex) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            index.dense_search([0.1, 0.2], limit=2)

    def test_metadata_filter_restricts_results_to_a_document(self, index: SearchIndex) -> None:
        allowed = index.allowed_positions(["D2"])
        results = index.dense_search(bag_of_words("goods delivery"), limit=5, allowed=allowed)

        assert results
        assert all(index.chunk_at(position).doc_id == "D2" for position, _ in results)

    def test_no_filter_searches_everything(self, index: SearchIndex) -> None:
        assert index.allowed_positions(None) is None

    def test_clear_empties_the_index(self, index: SearchIndex) -> None:
        index.clear()
        assert len(index) == 0
        assert not index
        assert index.dense_search(bag_of_words("anything"), limit=3) == []


class TestFusion:
    def test_rewards_agreement_between_rankings(self) -> None:
        dense = [(5, 0.9), (1, 0.8)]
        lexical = [(1, 12.0), (9, 3.0)]

        fused = reciprocal_rank_fusion([dense, lexical], rrf_k=60)

        # Position 1 is ranked by both lists, so it must outrank either leader.
        assert fused[1] > fused[5]
        assert fused[1] > fused[9]

    def test_handles_empty_rankings(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == {}


class TestMaximalMarginalRelevance:
    def test_prefers_relevance_when_lambda_is_one(self) -> None:
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        relevance = {0: 1.0, 1: 0.9, 2: 0.1}

        chosen = maximal_marginal_relevance([0, 1, 2], relevance, vectors, 2, 1.0)
        assert chosen == [0, 1]

    def test_breaks_up_near_duplicates_when_diversity_matters(self) -> None:
        # Positions 0 and 1 are identical vectors; 2 is orthogonal and less relevant.
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        relevance = {0: 1.0, 1: 0.9, 2: 0.5}

        chosen = maximal_marginal_relevance([0, 1, 2], relevance, vectors, 2, 0.5)
        assert chosen == [0, 2]

    def test_vectors_are_indexed_by_candidate_order_not_relevance_order(self) -> None:
        # `vectors` is built from `candidates`, so row 0 here is position 2.
        # Passing candidates in an order the relevance sort will permute catches
        # any lookup that keys the similarity matrix on the sorted order instead.
        candidates = [2, 0, 1]
        vectors = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        relevance = {0: 1.0, 1: 0.95, 2: 0.5}

        chosen = maximal_marginal_relevance(candidates, relevance, vectors, 2, 0.5)

        # 1 duplicates 0, so the diverse 2 must be preferred despite lower relevance.
        assert chosen == [0, 2]

    def test_falls_back_to_relevance_without_vectors(self) -> None:
        chosen = maximal_marginal_relevance([3, 1], {3: 0.2, 1: 0.9}, None, 2, 0.7)
        assert chosen == [1, 3]

    def test_handles_no_candidates(self) -> None:
        assert maximal_marginal_relevance([], {}, None, 3, 0.7) == []


class TestHybridRetriever:
    def test_retrieves_the_relevant_clause(self, index: SearchIndex) -> None:
        retriever = HybridRetriever(index, StubEmbedder(), RetrievalSettings(top_k=2))

        results = retriever.retrieve("How much notice is needed to terminate?")

        assert results
        assert "terminate" in results[0].chunk.text.lower()
        assert results[0].rank == 1

    def test_records_which_index_matched(self, index: SearchIndex) -> None:
        retriever = HybridRetriever(index, StubEmbedder(), RetrievalSettings(top_k=3))

        results = retriever.retrieve("penalty for late delivery of goods")

        top = results[0]
        assert top.matched_lexically or top.matched_semantically

    def test_scope_filter_limits_results_to_selected_documents(self, index: SearchIndex) -> None:
        retriever = HybridRetriever(index, StubEmbedder(), RetrievalSettings(top_k=5))

        results = retriever.retrieve("payment and delivery", doc_ids=["D2"])

        assert results
        assert {item.chunk.doc_id for item in results} == {"D2"}

    def test_unknown_document_id_yields_no_results(self, index: SearchIndex) -> None:
        retriever = HybridRetriever(index, StubEmbedder(), RetrievalSettings())
        assert retriever.retrieve("anything", doc_ids=["D999"]) == []

    def test_blank_query_returns_nothing(self, index: SearchIndex) -> None:
        retriever = HybridRetriever(index, StubEmbedder(), RetrievalSettings())
        assert retriever.retrieve("   ") == []

    def test_querying_an_empty_index_raises(self) -> None:
        retriever = HybridRetriever(SearchIndex(), StubEmbedder(), RetrievalSettings())
        with pytest.raises(RetrievalError):
            retriever.retrieve("termination")

    def test_never_returns_more_than_top_k(self, index: SearchIndex) -> None:
        retriever = HybridRetriever(index, StubEmbedder(), RetrievalSettings(top_k=2))
        assert len(retriever.retrieve("contract payment termination goods")) <= 2

    def test_ranks_are_dense_and_sequential(self, index: SearchIndex) -> None:
        retriever = HybridRetriever(index, StubEmbedder(), RetrievalSettings(top_k=3))
        results = retriever.retrieve("contract termination")

        assert [item.rank for item in results] == list(range(1, len(results) + 1))
