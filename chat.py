#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=1.0",
#     "python-dotenv>=1.0",
# ]
# ///
"""Simple terminal chat against OpenCode Go (Muse Spark 1.3 Contributor)."""

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://opencode.ai/zen/go/v1"
MODEL = os.environ.get("OPENCODEGO_MODEL", "deepseek-v4-flash")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    api_key = os.environ.get("OPENCODEGO_API_KEY")
    if not api_key:
        sys.exit("OPENCODEGO_API_KEY not found in ../.env")

    session_id = str(uuid.uuid4())
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        default_headers={"x-opencode-session": session_id},
    )
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    print(f"Chatting with {MODEL} (opencode go). Ctrl+C or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"ai> {reply}\n".encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))


if __name__ == "__main__":
    main()
