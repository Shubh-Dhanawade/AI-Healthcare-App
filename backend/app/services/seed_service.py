"""
Service to seed RAG query evaluation logs.
"""

from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.user import User
from app.models.rag_query_log import RAGQueryLog

SEED_DATA = [
    {
        "query": "Which is premium amount in this project?",
        "answer": "The annual premium for this policy is 12,500, including GST.",
        "faithfulness": 1.0,
        "faithfulness_reasoning": "The premium amount is directly mentioned as 12,500 including GST on Page 4 of Basic_Health_Insurance_Report.pdf.",
        "answer_relevance": 0.98,
        "answer_relevance_reasoning": "Perfect relevance. Directly answers the premium amount request.",
        "context_relevance": 0.92,
        "latency": 0.95
    },
    {
        "query": "What is the policy sum insured?",
        "answer": "The total sum insured under the Basic Health Insurance Policy is 5,00,000.",
        "faithfulness": 0.95,
        "faithfulness_reasoning": "The answer is fully grounded in the retrieved context, which states a sum insured of 5,00,000.",
        "answer_relevance": 0.90,
        "answer_relevance_reasoning": "Highly relevant answer outlining the exact coverage amount.",
        "context_relevance": 0.88,
        "latency": 1.10
    },
    {
        "query": "which is best policy",
        "answer": "This Basic Health Insurance Policy (HI-2026-001234) offers a sum insured of 5,00,000 with cashless hospitalization and coverage for various medical treatments. It includes benefits like in-patient hospitalization, ICU, surgery, maternity, and preventive health check-ups. The annual premium is 12,500, including GST, and it has 30-day pre/post-hospitalization waiting periods. I could not find that specific detail about which policy is 'best'.",
        "faithfulness": 0.98,
        "faithfulness_reasoning": "The model explains the benefits of HI-2026-001234 but correctly states there is no context to rank or compare which policy is 'best'.",
        "answer_relevance": 0.95,
        "answer_relevance_reasoning": "Appropriately addresses the subjective query given context limitations.",
        "context_relevance": 0.85,
        "latency": 1.45
    },
    {
        "query": "What are the pre-existing disease coverage rules?",
        "answer": "Pre-existing diseases are covered after a continuous waiting period of 36 months under the policy terms.",
        "faithfulness": 0.88,
        "faithfulness_reasoning": "The waiting period of 36 months for pre-existing conditions is explicitly stated on Page 2 of the policy documentation.",
        "answer_relevance": 0.85,
        "answer_relevance_reasoning": "Direct answer regarding the coverage waiting timelines.",
        "context_relevance": 0.80,
        "latency": 1.25
    },
    {
        "query": "Is maternity covered under the standard plan?",
        "answer": "Yes, maternity expenses are covered up to a limit of 50,000 for normal delivery and 75,000 for Caesarean section, subject to a waiting period of 24 months of continuous coverage.",
        "faithfulness": 0.92,
        "faithfulness_reasoning": "The details regarding maternity benefits and waiting period match the policy limits specified on Page 3.",
        "answer_relevance": 0.88,
        "answer_relevance_reasoning": "Direct and relevant explanation of maternity sub-limits.",
        "context_relevance": 0.84,
        "latency": 1.30
    },
    {
        "query": "What is the ICU room rent limit?",
        "answer": "ICU room rent is covered up to 2% of the sum insured per day under the policy.",
        "faithfulness": 1.0,
        "faithfulness_reasoning": "Direct mapping to Page 2 where the ICU limit is specified as 2% of the 5 Lakhs sum insured (10,000/day).",
        "answer_relevance": 0.92,
        "answer_relevance_reasoning": "Answers ICU charges directly with percentage details.",
        "context_relevance": 0.90,
        "latency": 0.85
    },
    {
        "query": "Are ambulance charges covered?",
        "answer": "Yes, emergency road ambulance charges are covered up to 2,000 per hospitalization.",
        "faithfulness": 0.96,
        "faithfulness_reasoning": "The coverage of 2,000 for emergency ambulance charges is fully backed by the policy text on Page 3.",
        "answer_relevance": 0.96,
        "answer_relevance_reasoning": "Clearly confirms coverage limits for ambulance service.",
        "context_relevance": 0.86,
        "latency": 0.75
    },
    {
        "query": "What is the pre-hospitalization coverage period?",
        "answer": "Pre-hospitalization expenses are covered for a period of up to 30 days prior to admission.",
        "faithfulness": 0.90,
        "faithfulness_reasoning": "The 30-day pre-hospitalization limit is documented on Page 1 of the terms.",
        "answer_relevance": 0.90,
        "answer_relevance_reasoning": "Identifies the correct 30-day duration.",
        "context_relevance": 0.82,
        "latency": 0.90
    },
    {
        "query": "What is the post-hospitalization coverage period?",
        "answer": "Post-hospitalization medical expenses are covered for up to 60 days after discharge.",
        "faithfulness": 0.95,
        "faithfulness_reasoning": "The 60-day post-hospitalization limit is explicitly grounded on Page 1.",
        "answer_relevance": 0.92,
        "answer_relevance_reasoning": "Directly references the 60-day post-hospitalization benefit.",
        "context_relevance": 0.85,
        "latency": 1.05
    },
    {
        "query": "Does the policy cover dental treatments?",
        "answer": "Dental treatment is excluded from the standard policy unless it is required due to an accidental injury and necessitates hospitalization.",
        "faithfulness": 1.0,
        "faithfulness_reasoning": "Grounding verified. Dental care is generally listed under exclusions, except for accidental trauma hospitalization.",
        "answer_relevance": 0.95,
        "answer_relevance_reasoning": "Provides nuance about exclusions vs accidental exceptions.",
        "context_relevance": 0.88,
        "latency": 1.20
    },
    {
        "query": "Is cosmetic surgery covered?",
        "answer": "No, cosmetic or aesthetic treatments are completely excluded from the policy coverage, unless necessitated by an accident.",
        "faithfulness": 0.85,
        "faithfulness_reasoning": "Grounding confirmed in the General Exclusions section on Page 5.",
        "answer_relevance": 0.80,
        "answer_relevance_reasoning": "Directly addresses aesthetic treatment rules.",
        "context_relevance": 0.78,
        "latency": 1.15
    },
    {
        "query": "What is the co-pay for senior citizens?",
        "answer": "Co-payment of 10% is applicable for insured individuals above 60 years of age.",
        "faithfulness": 0.94,
        "faithfulness_reasoning": "Grounded in policy clause 4.2; however, the model did not mention the optional co-pay waiver add-on.",
        "answer_relevance": 0.90,
        "answer_relevance_reasoning": "Mentions the 10% co-payment rate for senior citizens correctly.",
        "context_relevance": 0.87,
        "latency": 1.40
    },
    {
        "query": "How do I claim cashless hospitalization?",
        "answer": "For cashless claims, you must submit a pre-authorization form at a network hospital at least 48 hours prior to planned treatment, or within 24 hours of emergency admission.",
        "faithfulness": 0.92,
        "faithfulness_reasoning": "The pre-authorization timelines match the claims guide section of the document.",
        "answer_relevance": 0.89,
        "answer_relevance_reasoning": "Examines the pre-authorization timelines step by step.",
        "context_relevance": 0.85,
        "latency": 1.65
    },
    {
        "query": "Are psychiatric treatments covered?",
        "answer": "Yes, psychiatric and mental illness treatments are covered up to the sum insured, subject to standard hospitalization terms.",
        "faithfulness": 0.97,
        "faithfulness_reasoning": "Mental illness cover is included up to the sum insured, but the model omitted details on out-patient consultation limits.",
        "answer_relevance": 0.94,
        "answer_relevance_reasoning": "Addresses psychiatric benefits comprehensively.",
        "context_relevance": 0.91,
        "latency": 1.35
    },
    {
        "query": "What is the waiting period for cataracts?",
        "answer": "Cataract surgeries have a specific waiting period of 24 months from the policy start date.",
        "faithfulness": 0.90,
        "faithfulness_reasoning": "Correctly identifies the 24-month waiting period for cataract surgery from the specific exclusion list.",
        "answer_relevance": 0.85,
        "answer_relevance_reasoning": "Outlines the waiting period limit accurately.",
        "context_relevance": 0.83,
        "latency": 1.00
    },
    {
        "query": "Can I renew the policy lifelong?",
        "answer": "Yes, the policy offers lifelong renewability, subject to timely premium payments and no fraud.",
        "faithfulness": 1.0,
        "faithfulness_reasoning": "Renewability terms are grounded in Section 5.1 of the policy documents.",
        "answer_relevance": 1.0,
        "answer_relevance_reasoning": "Confirms lifelong renewability rules explicitly.",
        "context_relevance": 1.0,
        "latency": 1.48
    }
]

async def seed_initial_rag_logs(db: AsyncSession, default_user_id: str = None):
    """Seed RAGQueryLog table with high quality realistic queries if empty."""
    # Check if we already have records
    count_stmt = select(RAGQueryLog)
    count_res = await db.execute(count_stmt)
    existing_logs = count_res.scalars().all()
    if existing_logs:
        logger.info(f"Skipping seeding: RAGQueryLog table already has {len(existing_logs)} entries.")
        return len(existing_logs)

    # Find a user to link logs to
    user_id = default_user_id
    if not user_id:
        user_res = await db.execute(select(User).order_by(User.created_at.asc()).limit(1))
        first_user = user_res.scalar_one_or_none()
        if not first_user:
            # Create a fallback dummy user if database has absolutely no users
            dummy_user = User(
                email="admin@healthcare.com",
                full_name="System Admin",
                role="admin",
                hashed_password="hashed_dummy_password"
            )
            db.add(dummy_user)
            await db.flush()
            user_id = dummy_user.id
        else:
            user_id = first_user.id

    logger.info(f"Seeding RAG query evaluation logs for user_id: {user_id}")
    
    base_time = datetime.now(timezone.utc) - timedelta(days=2)
    for i, data in enumerate(SEED_DATA):
        log_time = base_time + timedelta(hours=i * 3)
        log = RAGQueryLog(
            user_id=user_id,
            query=data["query"],
            answer=data["answer"],
            faithfulness=data["faithfulness"],
            faithfulness_reasoning=data["faithfulness_reasoning"],
            answer_relevance=data["answer_relevance"],
            answer_relevance_reasoning=data["answer_relevance_reasoning"],
            context_relevance=data["context_relevance"],
            latency=data["latency"],
            created_at=log_time
        )
        db.add(log)
    
    await db.commit()
    logger.info("✅ Seeded 16 RAG query logs successfully.")
    return len(SEED_DATA)
