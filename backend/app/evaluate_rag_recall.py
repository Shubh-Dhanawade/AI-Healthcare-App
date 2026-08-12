"""
RAG Recall@5 Evaluation

WHY THIS SCRIPT EXISTS
-----------------------
Nothing in the current codebase computes Recall@5 (or any retrieval metric)
-- it's a number that exists only on the presentation slide. This script
computes it for real, using your actual retrieval function
(app.services.vector_store.search_vector_store).

WHAT RECALL@5 MEANS
---------------------
For a set of test queries where you already know which chunk of a document
SHOULD answer the question, Recall@5 = the fraction of those queries where
the correct chunk appeared anywhere in the top 5 retrieved results.

    Recall@5 = (queries where correct chunk was in top 5) / (total queries)

This measures RETRIEVAL quality specifically -- separate from whether the
LLM's final generated answer was good (that's what faithfulness/answer
relevance in RAGQueryLog already measure). A high Recall@5 means your
FAISS/hybrid search is finding the right source material; a low one means
the LLM may be generating from irrelevant or incomplete context, even if it
sounds fluent -- a real diagnostic for hallucination causes.

WHAT YOU NEED TO PROVIDE
--------------------------
A small labeled test set: for a document already in your database, write
5-15 realistic questions, and for each one, identify which chunk_index
actually contains the answer. The easiest way to find chunk_index values:
query your DocumentChunk table directly for a document you know well, or
run this against a document you just uploaded and skim its chunks.

USAGE
-----
Run from the backend/ directory (needs DB access), e.g.:
    python evaluate_rag_recall.py
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import AsyncSessionLocal  # adjust import if your session factory is named differently
from app.services.vector_store import search_vector_store
from app.models.document import Document, DocumentChunk
from sqlalchemy import select

TOP_K = 5

# ---------------------------------------------------------------------------
# FILL THIS IN: (document_id, query, correct_chunk_index) for each test question.
#
# To find chunk_index values for a document you know, run this once to list
# them (uncomment print_chunks_for_document call in main(), set a doc_id).
# ---------------------------------------------------------------------------
TEST_SET = [
    # {
    #     "document_id": "abc-123-doc-id",
    #     "query": "What is the waiting period for maternity benefits?",
    #     "correct_chunk_index": 4,
    # },
    # {
    #     "document_id": "abc-123-doc-id",
    #     "query": "Is ICU room rent covered?",
    #     "correct_chunk_index": 2,
    # },
]


async def print_chunks_for_document(db, document_id: str):
    """Utility: list all chunks for a document so you can find correct_chunk_index
    values when building TEST_SET. Call this manually, not part of the main eval."""
    stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id
    ).order_by(DocumentChunk.chunk_index)
    result = await db.execute(stmt)
    chunks = result.scalars().all()
    for c in chunks:
        preview = c.text_content[:120].replace("\n", " ")
        print(f"  chunk_index={c.chunk_index}: {preview}...")


async def evaluate_query(db, document_id: str, query: str, correct_chunk_index: int) -> dict:
    # search_vector_store expects a `policies` list of dicts describing which
    # documents to search over -- match its expected shape (adjust field names
    # if your actual call sites in chat_service.py build this differently).
    policies = [{"id": document_id}]

    retrieved = await search_vector_store(db, query, policies, top_k=TOP_K)

    retrieved_indices = []
    for r in retrieved:
        chunk_id = r.get("chunk_id")
        if chunk_id:
            # Query DocumentChunk by chunk_id from database to resolve correct index,
            # since vector_store search results do not include chunk_index directly.
            stmt = select(DocumentChunk.chunk_index).where(DocumentChunk.id == chunk_id)
            res = await db.execute(stmt)
            c_idx = res.scalar_one_or_none()
            if c_idx is not None:
                retrieved_indices.append(c_idx)

    hit = correct_chunk_index in retrieved_indices

    return {
        "query": query,
        "correct_chunk_index": correct_chunk_index,
        "retrieved_chunk_indices": retrieved_indices,
        "hit": hit,
    }


async def main():
    if not TEST_SET:
        print("TEST_SET is empty. Add at least 8-10 (document_id, query, "
              "correct_chunk_index) triples at the top of this script before running.")
        print("\nTip: to find chunk_index values for a document, add a call to "
              "print_chunks_for_document(db, '<your-doc-id>') below and run once.")
        return

    async with AsyncSessionLocal() as db:
        results = []
        for item in TEST_SET:
            result = await evaluate_query(
                db, item["document_id"], item["query"], item["correct_chunk_index"]
            )
            results.append(result)
            status = "HIT" if result["hit"] else "MISS"
            print(f"[{status}] \"{result['query']}\" "
                  f"(expected chunk {result['correct_chunk_index']}, "
                  f"got {result['retrieved_chunk_indices']})")

        hits = sum(1 for r in results if r["hit"])
        recall_at_5 = round((hits / len(results)) * 100, 1)

        print("\n" + "=" * 50)
        print(f"Recall@{TOP_K}: {recall_at_5}%  ({hits}/{len(results)} queries)")
        print("=" * 50)

        output = {
            "top_k": TOP_K,
            "total_queries": len(results),
            "hits": hits,
            "recall_at_5": recall_at_5,
            "per_query": results,
        }
        with open("rag_recall_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print("\nSaved to rag_recall_results.json")


if __name__ == "__main__":
    asyncio.run(main())
