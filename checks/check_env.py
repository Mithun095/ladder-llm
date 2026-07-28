import os
from dotenv import load_dotenv

load_dotenv()

assert os.environ.get("GROQ_API_KEY"), "GROQ_API_KEY missing from .env"
assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing from .env"
print("Both API keys present.")
