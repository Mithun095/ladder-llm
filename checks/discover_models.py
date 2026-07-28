import os

import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

print("=== Groq models ===")
groq_models = Groq(api_key=os.environ["GROQ_API_KEY"]).models.list()
for m in groq_models.data:
    print(m.id)

print("\n=== OpenRouter free models ===")
resp = requests.get("https://openrouter.ai/api/v1/models")
for m in resp.json()["data"]:
    if m["id"].endswith(":free"):
        print(m["id"])
