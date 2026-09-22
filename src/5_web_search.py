
# src/5_web_search.py

import os
from typing import Any, Dict, List

from tavily import TavilyClient
from dotenv import load_dotenv


# -----------------------------
# Environment
# -----------------------------

load_dotenv()


# -----------------------------
# Web search
# -----------------------------

def web_search(
    query: str,
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Search the web when local retrieved evidence
    is insufficient.

    Returns a list of normalized web results.
    """

    # Validate query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    # Validate result count
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= 20
    ):
        raise ValueError(
            "max_results must be an integer between 1 and 20."
        )

    # Get API key
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is missing from your .env file."
        )

    # Initialize client
    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=query.strip(),
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
        )

    except Exception as error:
        raise RuntimeError(
            f"Tavily web search failed: {error}"
        ) from error

    # Validate response
    if not isinstance(response, dict):
        raise RuntimeError(
            "Tavily returned an unexpected response."
        )

    results = []

    for item in response.get("results", []):

        if not isinstance(item, dict):
            continue

        title = item.get("title") or ""
        url = item.get("url") or ""
        content = item.get("content") or ""
        score = item.get("score", 0.0)

        # Skip results without usable evidence or URL
        if not url.strip() or not content.strip():
            continue

        # Validate score
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        results.append({
            "title": title.strip(),
            "url": url.strip(),
            "content": content.strip(),
            "score": score,
            "source": "web",
        })

    return results


# -----------------------------
# Standalone test
# -----------------------------

if __name__ == "__main__":

    query = input(
        "Enter your search query: "
    ).strip()

    try:
        results = web_search(query)

        print(f"\nTotal web results: {len(results)}")

        for index, result in enumerate(
            results,
            start=1
        ):
            print(f"\n--- Result {index} ---")
            print("Title:", result["title"])
            print("URL:", result["url"])
            print("Score:", result["score"])
            print("Content:", result["content"])

    except Exception as error:
        print(f"\nWeb search error: {error}")