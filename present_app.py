"""
present_app.py — Voice-demo presentation shell for the Self-Correcting RAG Agent project.

HOW TO USE
----------
1. Drop this file into the same folder as your graph.py (so the import at the
   top of the "Live Demo" section can find `rag_pipeline` and `REPO_IDENTIFIER`).
2. Install Streamlit in your existing venv:
       uv pip install streamlit
3. Run it:
       streamlit run present_app.py
4. It opens in your browser at localhost:8501. Record that browser window/tab
   with OBS (Display Capture or Window Capture on the browser window) — all
   your switching now happens via clicks inside one window, not alt-tab.

CUSTOMIZE
---------
- CODE_SNIPPETS below has placeholders. Either paste your real code in
  directly, or point it at your actual files (see `read_snippet()` — it will
  try to read the real file first and fall back to the placeholder text).
- MERMAID_GRAPH is already your real, current graph definition from our
  conversation — update it here if your graph structure changes again.
- The "Live Demo" tab imports your real `rag_pipeline` and runs it for real.
  Test your question once *before* recording so you know it'll complete
  cleanly within your Groq TPM budget.
"""

import time
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Self-Correcting RAG Agent", layout="wide", page_icon="🧭")

# ----------------------------------------------------------------------------
# CONFIG — adjust these to match your project
# ----------------------------------------------------------------------------
PROJECT_TITLE = "Self-Correcting RAG Agent"
PROJECT_SUBTITLE = "Adaptive routing, corrective retrieval, and hallucination-safe generation — built with LangGraph"

# Path to your real graph.py file (used by the "Code Walkthrough" tab to try to
# read real source instead of the placeholder strings below). Adjust if needed.
PROJECT_FILES = {
    "graph.py": "graph.py",
    "node.py": "node.py",
    "prompts.py": "prompts.py",
}


# Placeholder code shown if the real file can't be found next to this script.
CODE_SNIPPETS = {
    "Answer evaluation (routing logic)": '''\
@with_state_model(GraphState)
def answer_evaluation_node(state: GraphState):
    if state.retries >= MAX_RETRIES:
        return "max_generation_reached"

    hallucination_grade = hallucination_grader.invoke(
        {"documents": state.documents, "generation": state.generation}
    )

    if hallucination_grade.binary_score == "yes":
        answer_grade = answer_grader.invoke({
            "question": state.question,
            "generation": state.generation
        })
        return "useful" if answer_grade.binary_score == "yes" else "not relevant"
    else:
        return "hallucination"
''',
    "The state-decorator (fixes LangGraph's dict-passing)": '''\
def with_state_model(model_cls):
    def decorator(func):
        @wraps(func)
        def wrapper(state):
            if isinstance(state, dict):
                state = model_cls(**state)
            return func(state)
        return wrapper
    return decorator
''',
    "Web search — URL-scoped to the real repo (the fork-conflation fix)": '''\
@with_state_model(GraphState)
def web_search_node(state: GraphState):
    search_query = f'{state.rewritten_question} "{REPO_IDENTIFIER}"'
    new_docs = web_search_tool.invoke({"query": search_query})

    repo_path = REPO_IDENTIFIER.lower()
    relevant_docs = [
        d for d in new_docs
        if f"github.com/{repo_path}" in d.get("url", "").lower()
    ]

    web_results = [d["content"] for d in relevant_docs]
    updated_documents = state.documents + web_results
    return {"documents": updated_documents, "retrieval_num": state.retrieval_num + 1}
''',
}


def read_snippet(label: str, filename_hint: str = None) -> str:
    """Try to read real project source; fall back to the placeholder string."""
    if filename_hint:
        try:
            with open(filename_hint, "r", encoding="utf-8") as f:
                return f.read()
        except (FileNotFoundError, OSError):
            pass
    return CODE_SNIPPETS.get(label, "# snippet not found")


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title(PROJECT_TITLE)
st.caption(PROJECT_SUBTITLE)

tabs = st.tabs([
    "🏠 Overview",
    "🧭 Architecture",
    "🔀 Adaptive RAG",
    "✅ Corrective RAG",
    "🔁 Self-Correction",
    "▶️ Live Demo",
    "🔍 Case Study",
    "🛠️ Tech Stack",
    "🎓 Takeaways",
])

# ----------------------------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------------------------
with tabs[0]:
    st.header("Basic RAG trusts everything it's given")
    col1, col2 = st.columns(2)
    with col1:
        st.error("**Always retrieves** — even when the question needs no retrieval at all.")
        st.error("**Never checks documents** — irrelevant chunks go straight into the prompt.")
    with col2:
        st.error("**Never checks itself** — a hallucinated answer looks just as confident.")
        st.error("**No way out** — if retrieval fails, there's no honest fallback.")
    st.success(
        "This project addresses all four — with adaptive routing, corrective retrieval, "
        "and self-evaluation."
    )

# ----------------------------------------------------------------------------
# ARCHITECTURE
# ----------------------------------------------------------------------------
with tabs[1]:
    st.header("One graph, four retry/correction loops and three ways the graph can honestly exit")
    import os
    image_path = "architecture.png"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(
            f"Couldn't find `{image_path}` next to present_app.py. "
            "Save your graph diagram there, or update `image_path` above."
        )

# ----------------------------------------------------------------------------
# ADAPTIVE RAG
# ----------------------------------------------------------------------------
with tabs[2]:
    st.header("Adaptive RAG — route before you retrieve")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader(":blue[Vectorstore]")
        st.write("Question is about the ingested repo — rewrite the query and retrieve chunks.")
    with c2:
        st.subheader(":violet[Direct Answer]")
        st.write("Generic or conversational question — skip retrieval, answer straight from the LLM.")
    with c3:
        st.subheader(":orange[Web Search]")
        st.write("Question needs current or external info the repo can't provide.")
    st.info(
        "A structured-output LLM call (`Literal[\"vectorstore\", \"websearch\", \"QA_LM\"]`) "
        "classifies the question into exactly one route — a fixed label the graph's "
        "conditional edges branch on, not free-form text."
    )

# ----------------------------------------------------------------------------
# CORRECTIVE RAG
# ----------------------------------------------------------------------------
with tabs[3]:
    st.header("Corrective RAG — don't trust retrieval blindly")
    st.write("**Retrieve candidates → grade each document → keep only relevant ones**")
    st.write("If nothing survives the filter:")
    st.markdown(
        "1. **Retry the vectorstore** — up to a capped number of attempts, with a rewritten query\n"
        "2. **Fall back to web search** — once vectorstore retries are exhausted\n"
        "3. **Give up honestly** — once web search is also exhausted, no more silent guessing"
    )
    with st.expander("Show the relevance grader"):
        st.code(
            'retrieval_grader.batch([{"question": q, "document": d} for d in documents])\n'
            '# -> binary_score: "yes" / "no" per document',
            language="python",
        )

# ----------------------------------------------------------------------------
# SELF-CORRECTION
# ----------------------------------------------------------------------------
with tabs[4]:
    st.header("Self-correction — grade the answer, not just the docs")
    st.write("After generation, the answer is graded for **hallucination** and **relevance**:")
    o1, o2, o3, o4 = st.columns(4)
    o1.success("**Useful**\n\nGrounded and relevant — end the run.")
    o2.error("**Hallucination**\n\nNot grounded — generate feedback, retry.")
    o3.error("**Not Relevant**\n\nGrounded, but off-topic — rewrite query, retry.")
    o4.warning("**Max Retries**\n\nCapped attempts reached — give up honestly.")
    st.caption(
        "A shared retry counter increments on every generation attempt — both the "
        "hallucination loop and the not-relevant loop share the same cap."
    )

# ----------------------------------------------------------------------------
# LIVE DEMO — actually runs your real graph
# ----------------------------------------------------------------------------
with tabs[5]:
    st.header("Live run")
    st.caption(
        "This calls your real `rag_pipeline` from graph.py. Test your question once before "
        "recording so you know it completes cleanly within your Groq rate limit."
    )

    question = st.text_input(
        "Ask a question about the ingested repo",
        placeholder="e.g. How does this repo implement LoRA fine-tuning?",
    )
    run = st.button("Run", type="primary")

    if run and question:
        try:
            from graph import rag_pipeline  # noqa: import your real compiled graph
        except ImportError as e:
            st.error(
                f"Couldn't import `rag_pipeline` from graph.py ({e}). "
                "Make sure this file sits next to your graph.py."
            )
        else:
            trace_box = st.container()
            final_answer = None
            inputs = {"question": question}

            with st.status("Running the graph…", expanded=True) as status:
                for output in rag_pipeline.stream(inputs, stream_mode="updates"):
                    for node_name, value in output.items():
                        trace_box.markdown(f"**Node:** `{node_name}`")
                        if isinstance(value, dict) and "rewritten_question" in value:
                            label = "vectorstore rewrite" if node_name == "db_query_rewrite_node" else "websearch rewrite"
                            trace_box.caption(f"↳ {label}: {value['rewritten_question']}")
                        if isinstance(value, dict) and "generation" in value:
                            final_answer = value["generation"]
                        time.sleep(0.15)  # small pacing so the trace is watchable on recording
                status.update(label="Done", state="complete")

            st.subheader("Answer")
            st.write(final_answer or "_No generation produced — check the trace above._")

# ----------------------------------------------------------------------------
# CASE STUDY
# ----------------------------------------------------------------------------
with tabs[6]:
    st.header("Catching a subtle hallucination: source conflation")
    st.code('"How does this repo implement LoRA fine-tuning?"  — asked against nanoGPT, which has no LoRA support at all.')
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("#### 🔴 Before")
        st.write(
            "Web search pulled in forks like `nanoGPT-LoRA` and `minOFT`. The generator "
            "confidently described `get_lora_model()` and `lora_rank` as native to the repo."
        )
        st.caption("Confidently wrong — the hallucination grader missed it, because the claim "
                   "was grounded in real (mis-sourced) text.")
    with b2:
        st.markdown("#### 🟢 After the fix")
        st.write(
            "Web search results are filtered by URL — only pages actually hosted on the "
            "target repo count as evidence. Every fork and derivative gets discarded."
        )
        st.caption('"I couldn\'t find enough information to explain how that repository '
                   'implements LoRA fine-tuning."')

    with st.expander("Show the fix"):
        st.code(read_snippet("Web search — URL-scoped to the real repo (the fork-conflation fix)"),
                 language="python")

# ----------------------------------------------------------------------------
# TECH STACK
# ----------------------------------------------------------------------------
with tabs[7]:
    st.header("Tech stack")
    stack = [
        ("LangGraph", "Stateful graph orchestration, conditional routing, retry loops"),
        ("Chroma", "Local vector store for embedded code chunks"),
        ("HuggingFace Embeddings", "sentence-transformers/all-MiniLM-L6-v2 — free, local, no API key"),
        ("Groq (gpt-oss-20b)", "Fast open-weight LLM for generation, grading, and routing"),
        ("Tavily", "Live web search fallback when the vectorstore comes up empty"),
        ("GitLoader + RecursiveCharacterTextSplitter", "Repo ingestion and language-aware code chunking"),
    ]
    cols = st.columns(3)
    for i, (name, desc) in enumerate(stack):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.caption(desc)

# ----------------------------------------------------------------------------
# TAKEAWAYS
# ----------------------------------------------------------------------------
with tabs[8]:
    st.header("Key takeaways")
    st.markdown(
        "1. **State is dict-like at runtime** — Pydantic models define shape, but LangGraph "
        "passes plain dicts between nodes; handled with a small decorator.\n"
        "2. **One conditional-edges call per source node** — a second call silently overwrites "
        "the first; caught by visualizing the graph, not just reading the code.\n"
        "3. **Structured output for decisions, plain text for content** — anything the code "
        "branches on uses `with_structured_output`; anything a human reads doesn't.\n"
        "4. **Grounded isn't the same as correct** — a hallucination grader can pass an answer "
        "that's technically grounded, just in the wrong source. Verification has to reach "
        "all the way to provenance."
    )
