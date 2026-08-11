import urllib.request
import json

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/", timeout=3) as response:
        html = response.read().decode('utf-8')
        print("Backend Response:")
        print(json.dumps(json.loads(html), indent=2))
except Exception as e:
    print(f"Failed to connect to backend: {e}")
