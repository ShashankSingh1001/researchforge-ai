import json
import pytest
from backend.agents.state import ResearchState, initial_state
from backend.agents import planner, researcher, verifier, writer, critic, graph
from backend.agents.llm_client import LLMError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state():
    # returns a clean initial state for each test
    return initial_state(query="Impact of LLMs on Indian fintech", depth="quick", domain="finance")


@pytest.fixture
def state_with_subtasks(base_state):
    # state pre-populated with subtasks for researcher/verifier/writer/critic tests
    base_state["subtasks"] = [
        "What are LLM applications in Indian fintech?",
        "What are the risks of LLMs in Indian fintech?",
    ]
    return base_state


@pytest.fixture
def state_with_raw_results(state_with_subtasks):
    # state pre-populated with raw results for verifier/writer/critic tests
    state_with_subtasks["raw_results"] = [
        {
            "subtask": "What are LLM applications in Indian fintech?",
            "url": "https://example.gov/llm-fintech",
            "title": "LLMs in Fintech - Gov Report",
            "content": "Large language models are transforming Indian fintech through fraud detection, customer support automation, and credit risk assessment. Several major banks have deployed LLM-based solutions." * 3,
            "source_type": "web",
        },
        {
            "subtask": "What are LLM applications in Indian fintech?",
            "url": "https://example.com/llm-finance",
            "title": "LLMs Reshape Finance",
            "content": "Indian fintech startups are leveraging LLMs to build multilingual customer support tools and automate KYC processes. Adoption is growing rapidly across the sector." * 3,
            "source_type": "web",
        },
        {
            "subtask": "What are the risks of LLMs in Indian fintech?",
            "url": "https://example.org/risks",
            "title": "Risks of AI in Finance",
            "content": "The deployment of LLMs in financial services poses risks including hallucination, data privacy concerns, and regulatory uncertainty. Indian regulators are evaluating frameworks." * 3,
            "source_type": "web",
        },
    ]
    return state_with_subtasks


@pytest.fixture
def state_with_verified_results(state_with_raw_results):
    # state pre-populated with verified results for writer/critic tests
    state_with_raw_results["verified_results"] = [
        {**r, "credibility_score": 0.9, "has_contradictions": False, "contradictions": []}
        for r in state_with_raw_results["raw_results"]
    ]
    return state_with_raw_results


@pytest.fixture
def state_with_report(state_with_verified_results):
    # state pre-populated with a report for critic tests
    state_with_verified_results["report"] = """## 1. Overview
LLMs are transforming Indian fintech.
## 2. Key Insights
- Fraud detection improved by 60%.
## 3. Data and Statistics
No concrete statistics available.
## 4. Contradictions
No contradictions detected.
## 5. Sources
- LLMs in Fintech | https://example.gov/llm-fintech
## 6. Conclusion
LLMs will continue shaping Indian fintech."""
    return state_with_verified_results


# ---------------------------------------------------------------------------
# state.py tests
# ---------------------------------------------------------------------------

class TestState:
    def test_initial_state_defaults(self):
        # verifies default values are set correctly
        s = initial_state("test query")
        assert s["query"] == "test query"
        assert s["depth"] == "quick"
        assert s["domain"] == "general"
        assert s["subtasks"] == []
        assert s["raw_results"] == []
        assert s["verified_results"] == []
        assert s["report"] == ""
        assert s["critique"] == {}
        assert s["errors"] == []

    def test_initial_state_custom_values(self):
        # verifies custom depth and domain are respected
        s = initial_state("query", depth="detailed", domain="finance")
        assert s["depth"] == "detailed"
        assert s["domain"] == "finance"


# ---------------------------------------------------------------------------
# planner.py tests
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_planner_happy_path(self, base_state, monkeypatch):
        # verifies planner populates subtasks from valid LLM JSON response
        mock_response = json.dumps({
            "subtasks": ["Subtask 1", "Subtask 2"],
            "depth": "quick",
            "domain": "finance",
        })
        monkeypatch.setattr("backend.agents.planner.generate", lambda *a, **kw: mock_response)
        result = planner.run(base_state)
        assert result["subtasks"] == ["Subtask 1", "Subtask 2"]
        assert result["errors"] == []

    def test_planner_fallback_on_invalid_json(self, base_state, monkeypatch):
        # verifies planner falls back to original query when LLM returns invalid JSON
        monkeypatch.setattr("backend.agents.planner.generate", lambda *a, **kw: "not json")
        result = planner.run(base_state)
        assert result["subtasks"] == [base_state["query"]]
        assert len(result["errors"]) == 1

    def test_planner_fallback_on_llm_error(self, base_state, monkeypatch):
        # verifies planner handles LLMError gracefully
        monkeypatch.setattr("backend.agents.planner.generate", lambda *a, **kw: (_ for _ in ()).throw(LLMError("API down")))
        result = planner.run(base_state)
        assert result["subtasks"] == [base_state["query"]]
        assert len(result["errors"]) == 1

    def test_planner_enforces_max_subtasks(self, base_state, monkeypatch):
        # verifies planner trims subtasks exceeding MAX_SUBTASKS
        many_subtasks = [f"Subtask {i}" for i in range(20)]
        mock_response = json.dumps({"subtasks": many_subtasks, "depth": "quick", "domain": "finance"})
        monkeypatch.setattr("backend.agents.planner.generate", lambda *a, **kw: mock_response)
        result = planner.run(base_state)
        assert len(result["subtasks"]) <= 6

    def test_planner_strips_markdown_fences(self, base_state, monkeypatch):
        # verifies planner handles LLM responses wrapped in markdown code blocks
        mock_response = "```json\n" + json.dumps({"subtasks": ["Task 1"], "depth": "quick", "domain": "finance"}) + "\n```"
        monkeypatch.setattr("backend.agents.planner.generate", lambda *a, **kw: mock_response)
        result = planner.run(base_state)
        assert result["subtasks"] == ["Task 1"]


# ---------------------------------------------------------------------------
# researcher.py tests
# ---------------------------------------------------------------------------

class TestResearcher:
    def test_researcher_happy_path(self, state_with_subtasks, monkeypatch):
        # verifies researcher populates raw_results for each subtask
        mock_search = lambda q, max_results=2: [
            {"title": "Result 1", "url": "https://example.com/1", "snippet": "snippet 1"},
            {"title": "Result 2", "url": "https://example.com/2", "snippet": "snippet 2"},
        ]
        mock_scrape = lambda url: "scraped content " * 50
        monkeypatch.setattr("backend.agents.researcher.search", mock_search)
        monkeypatch.setattr("backend.agents.researcher.scrape", mock_scrape)
        result = researcher.run(state_with_subtasks)
        assert len(result["raw_results"]) == 4  # 2 subtasks x 2 results
        assert result["raw_results"][0]["url"] == "https://example.com/1"

    def test_researcher_falls_back_to_snippet_on_scrape_error(self, state_with_subtasks, monkeypatch):
        # verifies researcher uses snippet when scraping fails
        from backend.tools.scraper import ScrapeError
        mock_search = lambda q, max_results=2: [
            {"title": "Result", "url": "https://example.com/1", "snippet": "fallback snippet"},
        ]
        monkeypatch.setattr("backend.agents.researcher.search", mock_search)
        monkeypatch.setattr("backend.agents.researcher.scrape", lambda url: (_ for _ in ()).throw(ScrapeError("failed")))
        result = researcher.run(state_with_subtasks)
        assert result["raw_results"][0]["content"] == "fallback snippet"
        assert len(result["errors"]) > 0

    def test_researcher_handles_search_error(self, state_with_subtasks, monkeypatch):
        # verifies researcher logs error and continues when search fails
        from backend.tools.search_tool import SearchError
        monkeypatch.setattr("backend.agents.researcher.search", lambda *a, **kw: (_ for _ in ()).throw(SearchError("no results")))
        result = researcher.run(state_with_subtasks)
        assert result["raw_results"] == []
        assert len(result["errors"]) > 0

    def test_researcher_routes_pdf_urls(self, state_with_subtasks, monkeypatch):
        # verifies PDF URLs are routed to pdf_parser instead of scraper
        mock_search = lambda q, max_results=2: [
            {"title": "PDF Doc", "url": "https://example.com/report.pdf", "snippet": "pdf snippet"},
        ]
        mock_parse_pdf = lambda url: ["chunk one", "chunk two"]
        monkeypatch.setattr("backend.agents.researcher.search", mock_search)
        monkeypatch.setattr("backend.agents.researcher.parse_pdf", mock_parse_pdf)
        result = researcher.run(state_with_subtasks)
        assert result["raw_results"][0]["source_type"] == "pdf"

    def test_researcher_respects_depth_detailed(self, state_with_subtasks, monkeypatch):
        # verifies detailed depth requests more results per subtask
        state_with_subtasks["depth"] = "detailed"
        call_args = {}
        def mock_search(q, max_results=2):
            call_args["max_results"] = max_results
            return []
        monkeypatch.setattr("backend.agents.researcher.search", mock_search)
        researcher.run(state_with_subtasks)
        assert call_args["max_results"] == 5


# ---------------------------------------------------------------------------
# verifier.py tests
# ---------------------------------------------------------------------------

class TestVerifier:
    def test_verifier_deduplicates_by_url(self, state_with_subtasks, monkeypatch):
        # verifies duplicate URLs are removed keeping only first occurrence
        monkeypatch.setattr("backend.agents.verifier.generate", lambda *a, **kw: json.dumps({"contradictions": [], "has_contradictions": False}))
        state_with_subtasks["raw_results"] = [
            {"url": "https://example.com/1", "title": "A", "content": "content " * 20, "source_type": "web"},
            {"url": "https://example.com/1", "title": "A duplicate", "content": "content " * 20, "source_type": "web"},
            {"url": "https://example.com/2", "title": "B", "content": "content " * 20, "source_type": "web"},
        ]
        result = verifier.run(state_with_subtasks)
        urls = [r["url"] for r in result["verified_results"]]
        assert len(urls) == len(set(urls))

    def test_verifier_filters_short_content(self, state_with_subtasks, monkeypatch):
        # verifies results with content below minimum length are removed
        monkeypatch.setattr("backend.agents.verifier.generate", lambda *a, **kw: json.dumps({"contradictions": [], "has_contradictions": False}))
        state_with_subtasks["raw_results"] = [
            {"url": "https://example.com/1", "title": "Short", "content": "too short", "source_type": "web"},
            {"url": "https://example.com/2", "title": "Long", "content": "valid content " * 20, "source_type": "web"},
        ]
        result = verifier.run(state_with_subtasks)
        assert all(len(r["content"]) >= 100 for r in result["verified_results"])

    def test_verifier_scores_gov_domain_highest(self, state_with_subtasks, monkeypatch):
        # verifies .gov domains receive the highest credibility score
        monkeypatch.setattr("backend.agents.verifier.generate", lambda *a, **kw: json.dumps({"contradictions": [], "has_contradictions": False}))
        state_with_subtasks["raw_results"] = [
            {"url": "https://example.com/page", "title": "Com", "content": "content " * 20, "source_type": "web"},
            {"url": "https://treasury.gov/report", "title": "Gov", "content": "content " * 20, "source_type": "web"},
        ]
        result = verifier.run(state_with_subtasks)
        scores = {r["url"]: r["credibility_score"] for r in result["verified_results"]}
        assert scores["https://treasury.gov/report"] > scores["https://example.com/page"]

    def test_verifier_handles_llm_error_gracefully(self, state_with_raw_results, monkeypatch):
        # verifies contradiction detection failure does not crash the pipeline
        monkeypatch.setattr("backend.agents.verifier.generate", lambda *a, **kw: (_ for _ in ()).throw(LLMError("API down")))
        result = verifier.run(state_with_raw_results)
        assert len(result["verified_results"]) > 0


# ---------------------------------------------------------------------------
# writer.py tests
# ---------------------------------------------------------------------------

class TestWriter:
    def test_writer_happy_path(self, state_with_verified_results, monkeypatch):
        # verifies writer populates report from LLM response
        monkeypatch.setattr("backend.agents.writer.generate", lambda *a, **kw: "## 1. Overview\nTest report.")
        result = writer.run(state_with_verified_results)
        assert result["report"] == "## 1. Overview\nTest report."
        assert result["errors"] == []

    def test_writer_handles_empty_verified_results(self, base_state, monkeypatch):
        # verifies writer logs error and sets empty report when no verified results exist
        result = writer.run(base_state)
        assert result["report"] == ""
        assert len(result["errors"]) == 1

    def test_writer_handles_llm_error(self, state_with_verified_results, monkeypatch):
        # verifies writer handles LLMError without crashing
        monkeypatch.setattr("backend.agents.writer.generate", lambda *a, **kw: (_ for _ in ()).throw(LLMError("API down")))
        result = writer.run(state_with_verified_results)
        assert result["report"] == ""
        assert len(result["errors"]) == 1


# ---------------------------------------------------------------------------
# critic.py tests
# ---------------------------------------------------------------------------

class TestCritic:
    def test_critic_happy_path(self, state_with_report, monkeypatch):
        # verifies critic populates critique dict from valid LLM JSON response
        mock_critique = {
            "quality_score": 7,
            "missing_insights": ["market size data"],
            "weak_citations": [],
            "logical_gaps": [],
            "suggestions": ["Add statistics"],
            "verdict": "acceptable",
        }
        monkeypatch.setattr("backend.agents.critic.generate", lambda *a, **kw: json.dumps(mock_critique))
        result = critic.run(state_with_report)
        assert result["critique"]["quality_score"] == 7
        assert result["critique"]["verdict"] == "acceptable"
        assert result["errors"] == []

    def test_critic_handles_empty_report(self, base_state):
        # verifies critic logs error and sets empty critique when no report exists
        result = critic.run(base_state)
        assert result["critique"] == {}
        assert len(result["errors"]) == 1

    def test_critic_handles_invalid_json(self, state_with_report, monkeypatch):
        # verifies critic returns default critique on malformed LLM response
        monkeypatch.setattr("backend.agents.critic.generate", lambda *a, **kw: "not json at all")
        result = critic.run(state_with_report)
        assert result["critique"]["verdict"] == "needs_revision"
        assert result["critique"]["quality_score"] == 0

    def test_critic_handles_llm_error(self, state_with_report, monkeypatch):
        # verifies critic handles LLMError without crashing
        monkeypatch.setattr("backend.agents.critic.generate", lambda *a, **kw: (_ for _ in ()).throw(LLMError("API down")))
        result = critic.run(state_with_report)
        assert result["critique"] == {}
        assert len(result["errors"]) == 1


# ---------------------------------------------------------------------------
# graph.py tests
# ---------------------------------------------------------------------------

class TestGraph:
    def test_run_pipeline_returns_research_state(self, monkeypatch):
        # verifies run_pipeline returns a complete ResearchState with all keys
        monkeypatch.setattr("backend.agents.planner.generate", lambda *a, **kw: json.dumps({"subtasks": ["subtask 1"], "depth": "quick", "domain": "finance"}))
        monkeypatch.setattr("backend.agents.researcher.search", lambda *a, **kw: [])
        monkeypatch.setattr("backend.agents.verifier.generate", lambda *a, **kw: json.dumps({"contradictions": [], "has_contradictions": False}))
        monkeypatch.setattr("backend.agents.writer.generate", lambda *a, **kw: "## 1. Overview\nTest.")
        monkeypatch.setattr("backend.agents.critic.generate", lambda *a, **kw: json.dumps({
            "quality_score": 6, "missing_insights": [], "weak_citations": [],
            "logical_gaps": [], "suggestions": [], "verdict": "acceptable"
        }))
        result = graph.run_pipeline("test query", depth="quick", domain="finance")
        assert "query" in result
        assert "report" in result
        assert "critique" in result
        assert "errors" in result

    def test_run_pipeline_propagates_query(self, monkeypatch):
        # verifies query is preserved correctly through the full pipeline
        monkeypatch.setattr("backend.agents.planner.generate", lambda *a, **kw: json.dumps({"subtasks": ["subtask 1"], "depth": "quick", "domain": "general"}))
        monkeypatch.setattr("backend.agents.researcher.search", lambda *a, **kw: [])
        monkeypatch.setattr("backend.agents.verifier.generate", lambda *a, **kw: json.dumps({"contradictions": [], "has_contradictions": False}))
        monkeypatch.setattr("backend.agents.writer.generate", lambda *a, **kw: "## 1. Overview\nTest.")
        monkeypatch.setattr("backend.agents.critic.generate", lambda *a, **kw: json.dumps({
            "quality_score": 6, "missing_insights": [], "weak_citations": [],
            "logical_gaps": [], "suggestions": [], "verdict": "acceptable"
        }))
        result = graph.run_pipeline("my specific query")
        assert result["query"] == "my specific query"