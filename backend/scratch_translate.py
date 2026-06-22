import asyncio
import os
import sys
import httpx
from loguru import logger

# Add backend directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock settings environment variables
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./healthcare_ai.db"

from app.core.config import settings

# Test Prompt
TEST_PROMPT_TEMPLATE = """You are an expert translator. Translate the English text into {target_language}.

Rules:
1. Translate to {target_language} language.
2. Use the correct script for {target_language}. For example, if target language is Marathi, use Devanagari script (like 'पॉलिसीचे नियम व अटी'). If Hindi, use Devanagari script (like 'पॉलिसी के नियम और शर्तें').
3. Do NOT mix or use letters from other scripts like Gujarati or Bengali.
4. Output ONLY the translation. Do NOT include any explanations, introductory text, or formatting.

Examples for Marathi (मराठी):
English: "This policy covers hospital room rent up to 1% of sum insured."
Marathi: "या पॉलिसीमध्ये विमा रक्कमेच्या १% पर्यंत रुग्णालयाच्या खोलीच्या भाड्याचा समावेश आहे."

English: "Pre-existing diseases are covered after a waiting period of 3 years."
Marathi: "३ वर्षांच्या प्रतीक्षा कालावधीनंतर आधीपासून असलेले आजार कव्हर केले जातात."

Examples for Hindi (हिंदी):
English: "This policy covers hospital room rent up to 1% of sum insured."
Hindi: "यह पॉलिसी बीमा राशि के १% तक अस्पताल के कमरे के किराए को कवर करती है।"

TEXT TO TRANSLATE:
{text}"""

async def call_ollama(prompt: str) -> str:
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1, # Keep it low for deterministic translation
        }
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]

async def main():
    text = "The policy summary explains deductibles and coverage options simply."
    
    print("Testing translation to Marathi with few-shot prompt...")
    prompt_marathi = TEST_PROMPT_TEMPLATE.format(text=text, target_language="Marathi")
    result_marathi = await call_ollama(prompt_marathi)
    
    print("Testing translation to Hindi with few-shot prompt...")
    prompt_hindi = TEST_PROMPT_TEMPLATE.format(text=text, target_language="Hindi")
    result_hindi = await call_ollama(prompt_hindi)
    
    # Save to a text file in UTF-8
    with open("scratch_output.txt", "w", encoding="utf-8") as f:
        f.write("=== MARATHI TRANSLATION ===\n")
        f.write(result_marathi.strip() + "\n\n")
        f.write("=== HINDI TRANSLATION ===\n")
        f.write(result_hindi.strip() + "\n")
        
    print("Saved translations to scratch_output.txt!")

if __name__ == "__main__":
    asyncio.run(main())
