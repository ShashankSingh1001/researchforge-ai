import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# resolve imports regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.search_tool import search, SearchError
from backend.tools.scraper import scrape, ScrapeError
from backend.tools.pdf_parser import parse_pdf, PDFParseError


# --------------------------------------------------------------------------- #
# search_tool tests
# --------------------------------------------------------------------------- #

class TestSearch:

    def _make_ddgs_result(self, title, href, body):
        # helper to build a fake DDGS result dict
        return {"title": title, "href": href, "body": body}

    @patch("backend.tools.search_tool.DDGS")
    def test_returns_normalized_results(self, mock_ddgs_cls):
        # verifies returned dicts have title, url, snippet keys
        fake_results = [
            self._make_ddgs_result("Result One", "https://example.com/1", "Snippet one."),
            self._make_ddgs_result("Result Two", "https://example.com/2", "Snippet two."),
        ]
        mock_ddgs_cls.return_value.__enter__.return_value.text.return_value = fake_results

        results = search("LLMs in fintech", max_results=2)

        assert len(results) == 2
        assert results[0] == {"title": "Result One", "url": "https://example.com/1", "snippet": "Snippet one."}
        assert results[1]["url"] == "https://example.com/2"

    @patch("backend.tools.search_tool.DDGS")
    def test_respects_max_results_argument(self, mock_ddgs_cls):
        # verifies max_results is forwarded to DDGS
        mock_instance = mock_ddgs_cls.return_value.__enter__.return_value
        mock_instance.text.return_value = []

        search("test query", max_results=7)

        mock_instance.text.assert_called_once_with("test query", max_results=7)

    @patch("backend.tools.search_tool.DDGS")
    def test_returns_empty_list_on_no_results(self, mock_ddgs_cls):
        # verifies empty list returned when DDGS finds nothing
        mock_ddgs_cls.return_value.__enter__.return_value.text.return_value = []
        results = search("obscure query with no results")
        assert results == []

    @patch("backend.tools.search_tool.DDGS")
    def test_raises_search_error_on_ddgs_failure(self, mock_ddgs_cls):
        # verifies SearchError is raised when DDGS throws
        mock_ddgs_cls.return_value.__enter__.return_value.text.side_effect = RuntimeError("network down")
        with pytest.raises(SearchError, match="DuckDuckGo search failed"):
            search("any query")

    def test_raises_value_error_on_empty_query(self):
        # verifies empty/blank query raises ValueError before network call
        with pytest.raises(ValueError):
            search("")
        with pytest.raises(ValueError):
            search("   ")


# --------------------------------------------------------------------------- #
# scraper tests
# --------------------------------------------------------------------------- #

class TestScrape:

    def _make_response(self, text, status_code=200):
        # builds a minimal fake requests.Response
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = text
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @patch("backend.tools.scraper.requests.get")
    def test_returns_clean_text(self, mock_get):
        # verifies noise tags are stripped and clean text is returned
        html = """
        <html><body>
            <nav>Skip this nav</nav>
            <h1>Main Title</h1>
            <p>Useful paragraph content here.</p>
            <footer>Skip this footer</footer>
            <script>var x = 1;</script>
        </body></html>
        """
        mock_get.return_value = self._make_response(html)
        result = scrape("https://example.com")

        assert "Main Title" in result
        assert "Useful paragraph content here." in result
        assert "Skip this nav" not in result
        assert "Skip this footer" not in result
        assert "var x = 1" not in result

    @patch("backend.tools.scraper.requests.get")
    def test_raises_scrape_error_on_request_failure(self, mock_get):
        # verifies ScrapeError is raised when requests throws
        import requests as req
        mock_get.side_effect = req.RequestException("connection refused")
        with pytest.raises(ScrapeError, match="Failed to fetch URL"):
            scrape("https://unreachable.example.com")

    @patch("backend.tools.scraper.requests.get")
    def test_raises_scrape_error_on_http_error(self, mock_get):
        # verifies ScrapeError is raised on non-200 HTTP status
        import requests as req
        mock_resp = self._make_response("", status_code=404)
        mock_resp.raise_for_status.side_effect = req.HTTPError("404")
        mock_get.return_value = mock_resp
        with pytest.raises(ScrapeError, match="Failed to fetch URL"):
            scrape("https://example.com/missing")

    def test_raises_value_error_on_empty_url(self):
        # verifies empty/blank URL raises ValueError before network call
        with pytest.raises(ValueError):
            scrape("")
        with pytest.raises(ValueError):
            scrape("   ")

    @patch("backend.tools.scraper.requests.get")
    def test_uses_configured_timeout(self, mock_get):
        # verifies custom timeout is forwarded to requests.get
        mock_get.return_value = self._make_response("<html><body>ok</body></html>")
        scrape("https://example.com", timeout=25)
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 25


# --------------------------------------------------------------------------- #
# pdf_parser tests
# --------------------------------------------------------------------------- #

class TestParsePDF:

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        # creates a real single-page PDF using PyMuPDF for offline testing
        import fitz
        pdf_path = str(tmp_path / "test.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Artificial intelligence is transforming Indian fintech startups rapidly.")
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    @pytest.fixture
    def blank_pdf(self, tmp_path):
        # creates a real PDF with one blank page
        import fitz
        pdf_path = str(tmp_path / "blank.pdf")
        doc = fitz.open()
        doc.new_page()  # blank page, no text inserted
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def test_extracts_text_into_chunks(self, sample_pdf):
        # verifies text is extracted and returned as a non-empty list
        chunks = parse_pdf(sample_pdf)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert any("fintech" in chunk.lower() for chunk in chunks)

    def test_chunk_size_is_respected(self, sample_pdf):
        # verifies no chunk exceeds the configured word limit
        chunks = parse_pdf(sample_pdf, chunk_size=5)
        for chunk in chunks:
            assert len(chunk.split()) <= 5

    def test_blank_pdf_returns_empty_list(self, blank_pdf):
        # verifies blank PDF returns empty list without error
        result = parse_pdf(blank_pdf)
        assert result == []

    def test_raises_pdf_parse_error_on_bad_path(self):
        # verifies PDFParseError is raised for non-existent file
        with pytest.raises(PDFParseError, match="Failed to open PDF"):
            parse_pdf("/nonexistent/path/file.pdf")

    def test_raises_value_error_on_empty_path(self):
        # verifies empty path raises ValueError before any file access
        with pytest.raises(ValueError):
            parse_pdf("")
        with pytest.raises(ValueError):
            parse_pdf("   ")