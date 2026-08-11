import asyncio
import os
import sys

# Ensure backend folder is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ai_service import translate_text

async def main():
    test_phrases = [
        "Healthcare Policy Analysis Report",
        "No critical risk clauses detected.",
        "Document",
        "Processed"
    ]
    
    print("Testing translation to Hindi...")
    for phrase in test_phrases:
        res = await translate_text(phrase, "hindi")
        print(f"Original: {phrase} -> Hindi: {res}")
        
    print("\nTesting translation to Marathi...")
    for phrase in test_phrases:
        res = await translate_text(phrase, "marathi")
        print(f"Original: {phrase} -> Marathi: {res}")

if __name__ == "__main__":
    asyncio.run(main())
