import time
from langgraph.graph import StateGraph, END
from backend.agents.state import ResearchState, initial_state
from backend.agents import planner, researcher, verifier, writer, critic, evaluator
from backend.memory.vector_store import VectorStore, VectorStoreError, LONG_TERM

# shared LONG_TERM store persisted across sessions
_long_term_store = VectorStore()


def _build_graph() -> StateGraph:
    # constructs and compiles the LangGraph pipeline with all agent nodes
    graph = StateGraph(ResearchState)

    # register each agent as a named node
    graph.add_node("planner", planner.run)
    graph.add_node("researcher", researcher.run)
    graph.add_node("verifier", verifier.run)
    graph.add_node("writer", writer.run)
    graph.add_node("critic", critic.run)
    graph.add_node("evaluator", evaluator.run)

    # define linear execution order
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "verifier")
    graph.add_edge("verifier", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", "evaluator")
    graph.add_edge("evaluator", END)

    # planner is the entry point
    graph.set_entry_point("planner")

    return graph.compile()


# compiled graph instance, reused across all pipeline calls
_pipeline = _build_graph()


def _persist_to_long_term(state: ResearchState) -> None:
    # stores query, report, and source content in LONG_TERM memory for follow-up Q&A
    try:
        _long_term_store.store(
            state["query"],
            source=f"query|{state['domain']}",
            store_type=LONG_TERM,
        )

        if state["report"]:
            _long_term_store.store(
                state["report"],
                source=f"report|{state['query']}|{state['domain']}",
                store_type=LONG_TERM,
            )

        for result in state["verified_results"]:
            content = result.get("content", "")
            if content:
                _long_term_store.store(
                    content,
                    source=f"source|{result.get('url', '')}|{result.get('title', '')}",
                    store_type=LONG_TERM,
                )
    except VectorStoreError:
        # non-fatal: long-term persistence failure does not break the pipeline
        pass


def run_pipeline(query: str, depth: str = "quick", domain: str = "general") -> ResearchState:
    # executes full research pipeline, measures latency, persists to LONG_TERM memory
    state = initial_state(query=query, depth=depth, domain=domain)

    # start latency timer before pipeline execution
    start_time = time.time()
    result = _pipeline.invoke(state)
    latency = round(time.time() - start_time, 2)

    # inject latency into evaluation after pipeline completes
    result["evaluation"]["latency"] = latency

    _persist_to_long_term(result)
    return result


def run_followup(question: str, original_query: str = "") -> str:
    # answers a follow-up question using LONG_TERM memory without re-running pipeline
    try:
        chunks = _long_term_store.query(question, top_k=4, store_type=LONG_TERM)
    except VectorStoreError:
        return "Memory unavailable. Please run a research query first."

    if not chunks:
        return "No relevant context found in memory. Please run a research query first."

    context = ""
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")[:500]
        source = chunk.get("source", "memory")
        context += f"Context {i} ({source}):\n{text}\n\n"

    from backend.agents.llm_client import generate, LLMError

    prompt = f"""You are a research assistant. Answer the following question using only the provided context from previous research.

Original Research Query: {original_query}
Follow-up Question: {question}

Context:
{context}

Rules:
- Answer only from the provided context
- Be concise and direct
- If the context does not contain enough information, say so

Answer:"""

    try:
        return generate(prompt, max_new_tokens=512)
    except LLMError as e:
        return f"Failed to generate answer: {e}"