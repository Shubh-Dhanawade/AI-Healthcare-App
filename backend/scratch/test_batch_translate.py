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
        "Processed",
        "Pages",
        "AI Executive Summary",
        "Top Coverages",
        "Top Exclusions",
        "Extracted Policy Parameters",
        "Critical Risk Audit",
        "Clause",
        "Explanation",
        "Recommendation"
    ]
    
    separator = "\n----\n"
    combined_text = separator.join(test_phrases)
    
    try:
        translated_combined = await translate_text(combined_text, "hindi")
        translated_parts = [p.strip() for p in translated_combined.split("----")]
        
        output_path = "scratch/batch_results.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Results:\n")
            for orig, trans in zip(test_phrases, translated_parts):
                f.write(f"Original: {orig} -> Hindi: {trans}\n")
            f.write(f"\nLengths match: {len(test_phrases) == len(translated_parts)} ({len(test_phrases)} vs {len(translated_parts)})\n")
        print(f"Saved results to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
