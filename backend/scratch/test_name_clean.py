import re

candidates = [
    "Sangita Rajkumar Jain",
    "either at",
    "MISS Sangita Rajkumar Jain",
    "MISS Sangita Rajkumar Jain for the perio",
    "MISS Sangita Rajkumar Jain Thank you for",
    "of the Appointee (to Nominee)",
]

noise_regex = r'\b(thank|you|for|issued|to|per|the|period|either|at|inception|or|renewal|on|with|from|basis|policy|number|date|address|appointee|nominee|proposer|relation|relationship|member)\b.*$'

print("CLEANING RESULTS:")
for c in candidates:
    clean = re.sub(noise_regex, '', c, flags=re.IGNORECASE).strip()
    clean = re.sub(r'[,\s\-]+$', '', clean).strip() # Strip trailing punctuation
    print(f"Original: '{c}' -> Cleaned: '{clean}'")
