import requests
from bs4 import BeautifulSoup


# default headers to avoid basic bot detection
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# tags that add no research value
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form"]

DEFAULT_TIMEOUT = 10  # seconds


class ScrapeError(Exception):
    # raised when fetching or parsing a page fails
    pass


def scrape(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    # fetches URL and returns cleaned plain text content
    if not url or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ScrapeError(f"Failed to fetch URL '{url}': {e}") from e

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        raise ScrapeError(f"Failed to parse HTML from '{url}': {e}") from e

    # remove noise tags in-place
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # extract and clean visible text
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)