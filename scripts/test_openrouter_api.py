"""Verify the OpenRouter key works and each configured model returns a schema-valid trading signal,
using the same LLMClient the bot uses (so this tests the real code path, including fallback handling)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402  (also loads .env)
from src.llm import LLMClient, LLMUnavailable  # noqa: E402

PROMPT = (
    "Give a trading signal for a stock at $150 with 50-day MA $148, 200-day MA $145 and RSI 62. "
    "Reasoning must be one sentence."
)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("FAIL: OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    models = load_settings().models
    passed = 0
    for model in models:
        try:
            sig = LLMClient([model]).signal(PROMPT)
            print(f"OK   {model}\n     {sig.action} conf={sig.confidence} | {sig.reasoning}")
            passed += 1
        except LLMUnavailable as e:
            print(f"FAIL {model}: {e}")

    if passed == 0:
        print("\nFAIL: no configured model returned valid structured output.")
        print("Check the key, or that the model ids are still listed as free on openrouter.ai/models.")
        sys.exit(1)
    print(f"\n{passed}/{len(models)} models passed.")


if __name__ == "__main__":
    main()
