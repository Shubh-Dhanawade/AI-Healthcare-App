"""
OCR Accuracy Evaluation — Character Error Rate (CER) & Character Accuracy Rate (CAR)

WHY THIS SCRIPT EXISTS
-----------------------
Nothing in the current codebase computes OCR accuracy at all -- it's a number
that exists only on the presentation slide. This script computes it for real,
using your actual extraction pipeline (app.services.ocr_service).

HOW OCR ACCURACY IS MEASURED
------------------------------
CER (Character Error Rate) = edit_distance(reference, extracted) / len(reference)
    -- the standard OCR benchmark metric. Lower is better. It's the Levenshtein
    edit distance (insertions + deletions + substitutions) between the OCR
    output and the true text, divided by the reference length.

CAR (Character Accuracy Rate) = 1 - CER
    -- expressed as a percentage, this is what your "97% OCR accuracy" claim
    should actually be reporting.

WHAT YOU NEED TO PROVIDE
--------------------------
A small set of test documents where you already know the ground-truth text.
The easiest way to build this: take 5-8 of your sample policy PDFs/scanned
images, and for each one, manually type out (or copy-paste, for digital PDFs)
the CORRECT text for one representative page or section -- this becomes your
`reference` text. You do NOT need to transcribe the whole document, just a
consistent portion (e.g. the first page, or a specific clause) for a fair
comparison against what your OCR pipeline extracts for that same portion.

USAGE
-----
1. Fill in TEST_SET below with (file_path, reference_text) pairs.
2. pip install python-Levenshtein --break-system-packages   (fast C implementation)
3. Run from the backend/ directory:  python evaluate_ocr_accuracy.py
4. Paste the printed/JSON results into your dashboard or slide.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ocr_service import extract_document_text  # your actual pipeline

try:
    import Levenshtein
    def edit_distance(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except ImportError:
    # Fallback pure-Python implementation if python-Levenshtein isn't installed
    def edit_distance(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr_row = [i]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr_row.append(min(
                    prev_row[j] + 1,        # deletion
                    curr_row[j - 1] + 1,    # insertion
                    prev_row[j - 1] + cost  # substitution
                ))
            prev_row = curr_row
        return prev_row[-1]


# ---------------------------------------------------------------------------
# FILL THIS IN: (file_path, reference_text, file_type) for each test document.
# file_type must be "pdf" or "image" to match extract_document_text's signature.
# Keep reference_text focused on one page/section for a fair, manageable comparison.
# ---------------------------------------------------------------------------
TEST_SET = [
    # {
    #     "file_path": "/mnt/user-data/uploads/sample_policy_1.pdf",
    #     "file_type": "pdf",
    #     "reference_text": "PASTE THE GROUND-TRUTH TEXT OF PAGE 1 HERE...",
    # },
    # {
    #     "file_path": "/mnt/user-data/uploads/scanned_policy_2.png",
    #     "file_type": "image",
    #     "reference_text": "PASTE THE GROUND-TRUTH TEXT HERE...",
    # },
]


def normalize(text: str) -> str:
    """Basic normalization so formatting differences (extra spaces/newlines)
    don't unfairly inflate the error count. Keep this consistent between
    reference and extracted text."""
    return " ".join(text.split()).lower()


async def evaluate_document(file_path: str, file_type: str, reference_text: str) -> dict:
    extracted_text, method, page_count = await extract_document_text(file_path, file_type)

    ref_norm = normalize(reference_text)
    ext_norm = normalize(extracted_text)

    distance = edit_distance(ref_norm, ext_norm)
    cer = distance / max(len(ref_norm), 1)
    car = max(0.0, 1 - cer)  # clamp at 0 in the pathological case of a very bad extraction

    return {
        "file_path": file_path,
        "extraction_method": method,
        "reference_length": len(ref_norm),
        "extracted_length": len(ext_norm),
        "edit_distance": distance,
        "cer": round(cer * 100, 2),
        "car": round(car * 100, 2),
    }


async def main():
    if not TEST_SET:
        print("TEST_SET is empty. Add at least 3-5 (file_path, reference_text) "
              "pairs at the top of this script before running.")
        return

    results = []
    for item in TEST_SET:
        print(f"Evaluating: {item['file_path']} ...")
        result = await evaluate_document(
            item["file_path"], item["file_type"], item["reference_text"]
        )
        results.append(result)
        print(f"  -> method={result['extraction_method']}  "
              f"CER={result['cer']}%  CAR={result['car']}%")

    avg_cer = round(sum(r["cer"] for r in results) / len(results), 2)
    avg_car = round(sum(r["car"] for r in results) / len(results), 2)

    print("\n" + "=" * 50)
    print(f"Average CER (Character Error Rate):    {avg_cer}%")
    print(f"Average CAR (Character Accuracy Rate):  {avg_car}%   <-- this is your real 'OCR accuracy'")
    print("=" * 50)
    print(f"(computed on {len(results)} manually-verified documents)")

    output = {
        "documents_tested": len(results),
        "avg_cer": avg_cer,
        "avg_car": avg_car,
        "per_document": results,
    }
    with open("ocr_accuracy_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to ocr_accuracy_results.json")


if __name__ == "__main__":
    asyncio.run(main())
