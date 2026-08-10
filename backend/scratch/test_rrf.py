import asyncio
import re
import json
from typing import List, Dict, Any
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.services.vector_store import search_vector_store

_STOP_WORDS = {"the", "and", "for", "with", "that", "this", "what", "how", "are", "you", "can", "does", "did", "was", "has", "have"}

def test_text_search_fallback(query: str, policies: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
    query_lower = query.lower()
    SYNONYMS = {
        "dental": ["teeth", "tooth", "oral", "dentist", "dental treatment"],
        "maternity": ["pregnancy", "childbirth", "delivery", "newborn", "obstetric", "miscarriage"],
        "covered": ["covers", "coverage", "eligible", "payable", "admissible", "benefit"],
        "room rent": ["room charges", "accommodation", "hospital room", "icu", "shared room"],
        "pre-existing": ["pre existing", "ped", "pre-existing disease"],
        "waiting period": ["waiting", "moratorium", "initial period"],
        "premium": ["amount paid", "annual cost", "total premium"],
        "claim": ["reimbursement", "cashless", "hospital discharge"],
        "deductible": ["aggregate deductible", "excess", "deduct"],
        "ayush": ["ayurveda", "homeopathy", "unani", "siddha"],
        "ambulance": ["air ambulance", "transport", "emergency travel"],
        "cataract": ["eye", "lens", "vision"],
        "member": ["insured person", "covered person", "family member", "beneficiary",
                    "dependent", "self", "spouse", "son", "daughter", "insured persons",
                    "members covered", "persons covered", "member details", "insured members"],
        "who is covered": ["insured person", "covered person", "insured members", "member list",
                            "family members", "policy members", "persons insured"],
    }
    
    original_tokens = set(re.findall(r'\w+', query_lower))
    original_words = [w for w in original_tokens if len(w) > 2 and w not in _STOP_WORDS]
    
    synonym_words = set()
    for base_term, syns in SYNONYMS.items():
        if any(s in query_lower for s in [base_term] + syns):
            for s in syns:
                for w in s.split():
                    if w not in _STOP_WORDS and w not in original_words:
                        synonym_words.add(w)

    if not original_words:
        return []

    hits: List[Dict[str, Any]] = []

    for policy in policies:
        text = policy.get("text", "") or ""
        filename = policy.get("filename", "Policy")
        if not text:
            continue

        chunk_size = 1000
        overlap = 200
        start = 0
        chunks = []
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if end < len(text) and '\n' in text[end:end+100]:
                nl_pos = text.index('\n', end)
                chunk = text[start:nl_pos]
            chunks.append((start, chunk))
            start += chunk_size - overlap

        for pos, chunk in chunks:
            chunk_lower = chunk.lower()
            orig_score = sum(chunk_lower.count(w) * 5 for w in original_words)
            syn_score = sum(chunk_lower.count(w) for w in synonym_words)
            score = orig_score + syn_score
            
            if score > 0:
                schedule_boost = 0
                if any(k in chunk_lower for k in ["schedule of benefits", "section", "covered upto", "covered up to", "at actuals", "1.1", "1.2"]):
                    schedule_boost = 15 # boosted to match original word score scale
                if any(qw in chunk_lower for qw in original_words if len(qw) > 3):
                    schedule_boost += 10

                page_num = 1
                page_match = re.search(r'\[Page (\d+)\]', chunk)
                if page_match:
                    page_num = int(page_match.group(1))
                hits.append({
                    "text": chunk,
                    "source": filename,
                    "page": page_num,
                    "score": float(score + schedule_boost),
                    "document_id": policy.get("id", "")
                })

    hits.sort(key=lambda x: x["score"], reverse=True)
    seen_starts: set = set()
    deduped = []
    for h in hits:
        key = h["text"][:80].lower().strip()
        if key not in seen_starts:
            seen_starts.add(key)
            deduped.append(h)
    return deduped[:top_k]

async def main():
    target_doc_id = 'b65098e2-fb83-4cdb-8268-c93128a306c5'
    
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == target_doc_id).options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields)
        )
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    policies_data = [{
        "id": doc.id,
        "filename": doc.original_filename,
        "text": doc.extracted_text or "",
    }]
    
    query = "is this policy covers dental treatment"
    
    async with AsyncSessionLocal() as db:
        keyword_chunks = test_text_search_fallback(query, policies_data, top_k=8)
        vector_chunks = await search_vector_store(db, query, policies_data, top_k=8)
        
        # Reciprocal Rank Fusion
        rrf_scores = {}
        
        for rank, kc in enumerate(keyword_chunks):
            key = kc["text"][:80].lower().strip()
            rrf_scores[key] = {"chunk": kc, "k_rank": rank, "v_rank": None}
            
        for rank, vc in enumerate(vector_chunks):
            key = vc["text"][:80].lower().strip()
            if key not in rrf_scores:
                rrf_scores[key] = {"chunk": vc, "k_rank": None, "v_rank": rank}
            else:
                rrf_scores[key]["v_rank"] = rank

        combined_chunks = []
        for key, info in rrf_scores.items():
            k_score = 1.0 / (60 + info["k_rank"]) if info["k_rank"] is not None else 0.0
            v_score = 1.0 / (60 + info["v_rank"]) if info["v_rank"] is not None else 0.0
            info["chunk"]["rrf_score"] = k_score + v_score
            combined_chunks.append(info["chunk"])
            
        combined_chunks.sort(key=lambda x: x["rrf_score"], reverse=True)
        filtered_chunks = combined_chunks[:8]
        
    print("\n--- RETRIEVED CHUNKS WITH RRF ---")
    for idx, h in enumerate(filtered_chunks):
        safe_text = h['text'].replace('\n', ' ').encode('ascii', errors='replace').decode('ascii')
        k_rank_str = str(rrf_scores[h['text'][:80].lower().strip()]["k_rank"])
        v_rank_str = str(rrf_scores[h['text'][:80].lower().strip()]["v_rank"])
        print(f"Rank {idx+1} (RRF Score: {h['rrf_score']:.6f}, Keyword Rank: {k_rank_str}, Vector Rank: {v_rank_str}):")
        print(f"  {safe_text[:300]}...")

if __name__ == "__main__":
    asyncio.run(main())
