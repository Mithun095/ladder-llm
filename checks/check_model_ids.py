"""Validate every registry model ID against both providers' live catalogs.

Free-tier catalogs change without notice. A delisted model ID doesn't fail at import, in a
linter, or in a type check — it fails at runtime as a 404/400 on whichever query happens to
route to that tier, which is exactly how `poolside/laguna-m.1:free` was found (mid eval sweep,
days after it stopped existing).

Needs live network access and GROQ_API_KEY, so it isn't part of the CI job. Run it before
trusting a registry you haven't touched in a while.
"""
import os

import requests
from dotenv import load_dotenv
from groq import Groq

from src.registry import REGISTRY

load_dotenv()

groq_ids = {m.id for m in Groq(api_key=os.environ["GROQ_API_KEY"]).models.list().data}
openrouter = requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
openrouter_ids = {m["id"] for m in openrouter}

print(f"Live catalogs: {len(groq_ids)} Groq models, {len(openrouter_ids)} OpenRouter models.")

missing = []
for (tier, task_type), config in sorted(REGISTRY.items()):
    live = groq_ids if config.provider == "groq" else openrouter_ids
    if config.model_id not in live:
        missing.append((tier, task_type, config))

if missing:
    print("\nDELISTED MODEL IDS — the registry points at models that no longer exist:")
    for tier, task_type, config in missing:
        print(f"  tier {tier} / {task_type}: {config.provider} {config.model_id}")
    print("\nStill-available free OpenRouter models, for picking replacements:")
    for model_id in sorted(i for i in openrouter_ids if i.endswith(":free")):
        print(f"  {model_id}")
    raise SystemExit(f"{len(missing)} registry entries are stale.")

print(f"model id check passed: all {len(REGISTRY)} registry entries resolve to live models.")
