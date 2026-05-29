import os
import time
import io
import requests
import streamlit as st
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle

load_dotenv()

# base URL for FastAPI backend; override via .env BACKEND_URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Analyst",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (mirrors reference design system) ───────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

  :root {
    --bg:       #0a0c10;
    --surface:  #111318;
    --border:   #1e2330;
    --accent:   #00e5ff;
    --accent2:  #7c3aed;
    --warn:     #f59e0b;
    --danger:   #ef4444;
    --success:  #10b981;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --mono:     'DM Mono', monospace;
    --sans:     'Inter', sans-serif;
    --display:  'Syne', sans-serif;
  }

  html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; }
  * { font-family: var(--sans); box-sizing: border-box; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }
  .metric-card:hover { border-color: #2a3040; }
  .metric-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-family: var(--display);
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
  }
  .metric-sub {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.3rem;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
  }
  .section-header h2 {
    font-family: var(--display);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: var(--text);
    margin: 0;
  }
  .section-tag {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(0,229,255,0.08);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid rgba(0,229,255,0.2);
  }

  .hero-title {
    font-family: var(--display);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #fff 40%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
  }
  .hero-sub {
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-top: 0.4rem;
    text-transform: uppercase;
  }

  .pill {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
  }
  .pill-success { background: rgba(16,185,129,0.12);  color: var(--success); border: 1px solid rgba(16,185,129,0.25); }
  .pill-warn    { background: rgba(245,158,11,0.12);   color: var(--warn);    border: 1px solid rgba(245,158,11,0.25); }
  .pill-danger  { background: rgba(239,68,68,0.12);    color: var(--danger);  border: 1px solid rgba(239,68,68,0.25); }
  .pill-info    { background: rgba(0,229,255,0.08);    color: var(--accent);  border: 1px solid rgba(0,229,255,0.2); }
  .pill-muted   { background: rgba(100,116,139,0.12);  color: var(--muted);   border: 1px solid rgba(100,116,139,0.25); }

  .mono-block {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: #a8b3cf;
    line-height: 1.6;
  }

  .report-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    font-family: var(--sans);
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.8;
    white-space: pre-wrap;
  }

  /* styles for st.markdown rendered report content */
  .report-md h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.4rem !important;
    border-bottom: 1px solid #1e2330;
    padding-bottom: 0.3rem;
  }
  .report-md h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    color: #94a3b8 !important;
    margin-top: 1rem !important;
  }
  .report-md p, .report-md li {
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    line-height: 1.8 !important;
  }
  .report-md a { color: #00e5ff !important; }
  .report-md ul { padding-left: 1.25rem; }

  .agent-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
  }
  .agent-step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .agent-step-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .agent-step-status {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
  }

  .source-row {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
  }
  .source-title {
    font-family: var(--display);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.25rem;
  }
  .source-url {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--accent);
    margin-bottom: 0.5rem;
    word-break: break-all;
  }
  .source-snippet {
    font-family: var(--sans);
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.5;
  }

  .stButton > button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.5rem 1.5rem;
    border-radius: 6px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: rgba(0,229,255,0.08);
    border-color: var(--accent);
    color: var(--accent);
  }
  .stButton > button:active { transform: scale(0.98); }

  div[data-baseweb="select"] > div,
  div[data-baseweb="input"] > div,
  .stTextInput > div > div,
  .stTextArea textarea {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted) !important;
    background: transparent;
    border-radius: 5px;
    padding: 0.45rem 1.2rem;
    border: none;
  }
  .stTabs [aria-selected="true"] {
    background: var(--border) !important;
    color: var(--accent) !important;
  }

  .streamlit-expanderHeader {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted) !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
  }
  .streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
  }

  hr { border-color: var(--border); margin: 1.5rem 0; }
  .stProgress > div > div > div { background: linear-gradient(90deg, var(--accent2), var(--accent)); }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in [
    ("research_result", None),
    ("research_done", False),
    ("last_query", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ────────────────────────────────────────────────────────────────────

def _api_health() -> bool:
    # returns True if backend is reachable
    try:
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def _run_research(query: str, depth: str, domain: str) -> dict:
    # calls POST /api/research and returns the full response dict
    resp = requests.post(
        f"{BACKEND_URL}/api/research",
        json={"query": query, "depth": depth, "domain": domain},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _run_followup(question: str, original_query: str) -> dict:
    # calls POST /api/followup and returns the response dict
    resp = requests.post(
        f"{BACKEND_URL}/api/followup",
        json={"question": question, "original_query": original_query},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _credibility_pill(score: float) -> str:
    # returns HTML pill based on credibility score band
    if score >= 0.8:
        return f'<span class="pill pill-success">{score:.2f}</span>'
    if score >= 0.5:
        return f'<span class="pill pill-warn">{score:.2f}</span>'
    return f'<span class="pill pill-danger">{score:.2f}</span>'


def _score_color(value: float, invert: bool = False) -> str:
    # returns CSS color variable name based on a 0-1 score
    high = value >= 0.7
    if invert:
        high = not high
    return "var(--success)" if high else ("var(--warn)" if value >= 0.4 else "var(--danger)")


def _build_pdf(result: dict) -> bytes:
    # generates a print-ready PDF using black text on white — no dark-theme colors
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # all colors chosen for white paper readability
    BLACK      = colors.HexColor("#111111")
    DARK_GREY  = colors.HexColor("#333333")
    MID_GREY   = colors.HexColor("#555555")
    LIGHT_GREY = colors.HexColor("#888888")
    RULE_GREY  = colors.HexColor("#cccccc")
    HEADER_BG  = colors.HexColor("#1a1a2e")
    ROW_ALT    = colors.HexColor("#f5f5f5")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RATitle",
        parent=styles["Normal"],
        fontSize=22,
        textColor=BLACK,
        spaceAfter=4,
        fontName="Helvetica-Bold",
        leading=26,
    )
    sub_style = ParagraphStyle(
        "RASub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=LIGHT_GREY,
        spaceAfter=14,
        fontName="Helvetica",
    )
    section_style = ParagraphStyle(
        "RASection",
        parent=styles["Normal"],
        fontSize=11,
        textColor=BLACK,
        spaceBefore=14,
        spaceAfter=5,
        fontName="Helvetica-Bold",
        leading=14,
    )
    body_style = ParagraphStyle(
        "RABody",
        parent=styles["Normal"],
        fontSize=9,
        textColor=DARK_GREY,
        leading=15,
        fontName="Helvetica",
        spaceAfter=3,
    )
    bullet_style = ParagraphStyle(
        "RABullet",
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=2,
    )
    label_style = ParagraphStyle(
        "RALabel",
        parent=styles["Normal"],
        fontSize=8,
        textColor=MID_GREY,
        fontName="Helvetica",
        leading=11,
    )
    url_style = ParagraphStyle(
        "RAUrl",
        parent=styles["Normal"],
        fontSize=7,
        textColor=MID_GREY,
        fontName="Helvetica-Oblique",
        leading=10,
    )
    metric_key_style = ParagraphStyle(
        "RAMetricKey",
        parent=styles["Normal"],
        fontSize=8,
        textColor=DARK_GREY,
        fontName="Helvetica-Bold",
        leading=12,
    )
    metric_val_style = ParagraphStyle(
        "RAMetricVal",
        parent=styles["Normal"],
        fontSize=8,
        textColor=MID_GREY,
        fontName="Helvetica",
        leading=12,
    )
    tbl_header_style = ParagraphStyle(
        "RAThdr",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.white,
        fontName="Helvetica-Bold",
        leading=10,
    )

    story = []

    # title block
    story.append(Paragraph("Autonomous Research Analyst", title_style))
    story.append(Paragraph(f"Query: {result.get('query', '')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE_GREY))
    story.append(Spacer(1, 0.35 * cm))

    # parse report — handle ## headings and * /- bullets line by line
    report_text = result.get("report", "No report generated.")
    for line in report_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 0.1 * cm))
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(stripped.lstrip("# ").strip(), section_style))
        elif stripped.startswith("* ") or stripped.startswith("- "):
            story.append(Paragraph(f"\u2022 {stripped[2:].strip()}", bullet_style))
        else:
            story.append(Paragraph(stripped, body_style))

    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE_GREY))
    story.append(Spacer(1, 0.3 * cm))

    # verified sources table
    sources = result.get("sources", [])
    if sources:
        story.append(Paragraph("Verified Sources", section_style))
        story.append(Spacer(1, 0.2 * cm))
        table_data = [[
            Paragraph("Title",       tbl_header_style),
            Paragraph("URL",         tbl_header_style),
            Paragraph("Credibility", tbl_header_style),
        ]]
        for s in sources:
            table_data.append([
                Paragraph(s.get("title", "")[:80], label_style),
                Paragraph(s.get("url",   "")[:90], url_style),
                Paragraph(f"{s.get('credibility_score', 0):.2f}", label_style),
            ])
        tbl = Table(table_data, colWidths=[5.5 * cm, 9.5 * cm, 1.7 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  HEADER_BG),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, ROW_ALT]),
            ("GRID",          (0, 0), (-1, -1), 0.4, RULE_GREY),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

    # evaluation + critique metrics
    evaluation = result.get("evaluation", {})
    critique   = result.get("critique", {})
    if evaluation or critique:
        story.append(HRFlowable(width="100%", thickness=0.5, color=RULE_GREY))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Evaluation Metrics", section_style))
        story.append(Spacer(1, 0.15 * cm))

        ev_display = {}
        if evaluation:
            ev_display["Latency"]             = f"{evaluation.get('latency', 0):.2f}s"
            ev_display["Citation Score"]      = str(evaluation.get("citation_score", "N/A"))
            ev_display["Factual Score"]       = f"{evaluation.get('factual_score', 'N/A')}/10"
            ev_display["Hallucination Score"] = str(evaluation.get("hallucination_score", "N/A"))
            if evaluation.get("factual_reasoning"):
                ev_display["Factual Reasoning"] = evaluation["factual_reasoning"]
            if evaluation.get("hallucinated_claims"):
                ev_display["Hallucinated Claims"] = "; ".join(evaluation["hallucinated_claims"])
        if critique:
            ev_display["Quality Score"] = f"{critique.get('quality_score', 'N/A')}/10"
            ev_display["Verdict"]       = str(critique.get("verdict", "N/A"))
            if critique.get("missing_insights"):
                ev_display["Missing Insights"] = "; ".join(critique["missing_insights"])
            if critique.get("suggestions"):
                ev_display["Suggestions"] = "; ".join(critique["suggestions"])

        for k, v in ev_display.items():
            row_tbl = Table(
                [[Paragraph(k, metric_key_style), Paragraph(str(v), metric_val_style)]],
                colWidths=[4.5 * cm, 12.2 * cm],
            )
            row_tbl.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW",     (0, 0), (-1, -1), 0.3, RULE_GREY),
            ]))
            story.append(row_tbl)

    doc.build(story)
    return buf.getvalue()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                  color:#fff;letter-spacing:-0.01em;">◆ Research Analyst</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                  letter-spacing:0.12em;text-transform:uppercase;color:#64748b;
                  margin-top:2px;">Autonomous Multi-Agent System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem;">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Overview", "Research", "Sources", "Evaluation", "Follow-up Q&A"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#64748b;margin-bottom:0.75rem;">System Status</div>', unsafe_allow_html=True)

    backend_ok = _api_health()
    statuses = [
        ("Backend API",  backend_ok,                              "Online",    "Offline"),
        ("Research",     st.session_state.research_done,          "Complete",  "Pending"),
        ("Sources",      bool(st.session_state.research_result and st.session_state.research_result.get("sources")), "Loaded", "Pending"),
        ("Memory",       st.session_state.research_done,          "Active",    "Inactive"),
    ]
    for label, active, ok, nok in statuses:
        if label == "Backend API":
            pill_cls = "pill-success" if active else "pill-danger"
        else:
            pill_cls = "pill-success" if active else "pill-warn"
        text = ok if active else nok
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:0.5rem 0;border-bottom:1px solid #1e2330;">
          <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#94a3b8;">{label}</span>
          <span class="pill {pill_cls}">{text}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.6rem;color:#334155;text-align:center;margin-top:1rem;">v1.0.0 · research_analyst core</div>', unsafe_allow_html=True)


# ── Page: Overview ─────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("""
    <div style="margin-bottom:2rem">
      <div class="hero-title">Autonomous Research<br>Analyst</div>
      <div class="hero-sub">Multi-Agent Pipeline · RAG · Verification · Memory</div>
    </div>
    """, unsafe_allow_html=True)

    # pipeline architecture
    st.markdown("""
    <div class="section-header">
      <h2>Pipeline Architecture</h2>
      <span class="section-tag">Core</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns([1, 0.06, 1, 0.06, 1, 0.06, 1, 0.06, 1, 0.06, 1, 0.06, 1])
    stages = [
        ("01", "Planner",    "Breaks query into subtasks, sets depth and domain"),
        ("02", "Researcher", "Tavily search + scraper + PDF parser per subtask"),
        ("03", "Verifier",   "Deduplication, domain scoring, contradiction detection"),
        ("04", "Writer",     "RAG-grounded 6-section report with cited sources"),
        ("05", "Critic",     "Quality score, gaps, weak citations, verdict"),
        ("06", "Evaluator",  "Factual, citation, hallucination scores + latency"),
        ("07", "Memory",     "LONG_TERM FAISS persistence for follow-up Q&A"),
    ]
    for i, (num, title, desc) in enumerate(stages):
        with cols[i * 2]:
            st.markdown(f"""
            <div class="metric-card" style="min-height:140px">
              <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                          color:#00e5ff;letter-spacing:0.1em;margin-bottom:0.5rem">{num}</div>
              <div style="font-family:'Syne',sans-serif;font-weight:700;
                          font-size:0.82rem;color:#e2e8f0;margin-bottom:0.4rem">{title}</div>
              <div style="font-family:'Inter',sans-serif;font-size:0.7rem;
                          color:#64748b;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)
        if i < len(stages) - 1:
            with cols[i * 2 + 1]:
                st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:140px;color:#1e2330;font-size:1.2rem">&#8594;</div>', unsafe_allow_html=True)

    # module map
    st.markdown("""
    <div class="section-header" style="margin-top:2rem">
      <h2>Module Map</h2>
      <span class="section-tag">Source</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    modules = [
        ("tools/", "Tool Layer", [
            "search_tool.py — Tavily search API",
            "scraper.py — static HTML fetch + clean",
            "pdf_parser.py — PyMuPDF chunked extraction",
        ]),
        ("rag/  memory/", "RAG + Memory", [
            "embedder.py — all-MiniLM-L6-v2 vectors",
            "retriever.py — FAISS Flat L2 index",
            "vector_store.py — SHORT/LONG term stores",
        ]),
        ("agents/", "Agent Layer", [
            "planner, researcher, verifier, writer",
            "critic, evaluator, memory_agent",
            "graph.py — LangGraph linear pipeline",
        ]),
    ]
    for col, (fname, title, bullets) in zip([c1, c2, c3], modules):
        with col:
            items = "".join(f'<li style="margin-bottom:4px">{b}</li>' for b in bullets)
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                          color:#00e5ff;margin-bottom:0.3rem">{fname}</div>
              <div style="font-family:'Syne',sans-serif;font-weight:700;
                          font-size:0.82rem;color:#e2e8f0;margin-bottom:0.6rem">{title}</div>
              <ul style="font-family:'Inter',sans-serif;font-size:0.72rem;
                         color:#64748b;line-height:1.6;padding-left:1rem;margin:0">
                {items}
              </ul>
            </div>""", unsafe_allow_html=True)

    # quick-start guide
    st.markdown("""
    <div class="section-header" style="margin-top:2rem">
      <h2>Quick Start</h2>
      <span class="section-tag">Guide</span>
    </div>
    <div class="mono-block">
  1 &#8594;  Research        Enter a query, pick depth and domain, run the pipeline<br>
  2 &#8594;  Sources         Inspect verified sources with credibility scores<br>
  3 &#8594;  Evaluation      Review factual, citation, and hallucination metrics<br>
  4 &#8594;  Follow-up Q&A   Ask follow-up questions answered from LONG_TERM memory<br>
  <br>
  <span style="color:#00e5ff">&#9670;  Backend</span>   Start FastAPI with: uvicorn backend.main:app --reload
    </div>
    """, unsafe_allow_html=True)

    # output format spec
    st.markdown("""
    <div class="section-header" style="margin-top:2rem">
      <h2>Report Output Format</h2>
      <span class="section-tag">Schema</span>
    </div>
    """, unsafe_allow_html=True)
    sections = [
        ("1", "Overview",         "Brief summary of the research query and scope"),
        ("2", "Key Insights",     "Core findings drawn from verified sources"),
        ("3", "Data & Statistics","Quantitative evidence and key figures"),
        ("4", "Contradictions",   "Conflicting claims detected across sources"),
        ("5", "Sources",          "Cited URLs with titles used in the report"),
        ("6", "Conclusion",       "Synthesised answer and recommended next steps"),
    ]
    rows = "".join(f"""
    <tr>
      <td style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#00e5ff;padding:6px 12px">{n}</td>
      <td style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#7c3aed;padding:6px 12px">{s}</td>
      <td style="font-family:'Inter',sans-serif;font-size:0.72rem;color:#64748b;padding:6px 12px">{d}</td>
    </tr>""" for n, s, d in sections)
    st.markdown(f"""
    <div style="background:#111318;border:1px solid #1e2330;border-radius:8px;overflow:hidden">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="border-bottom:1px solid #1e2330">
            <th style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.1em;
                       text-transform:uppercase;color:#334155;padding:8px 12px;text-align:left">#</th>
            <th style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.1em;
                       text-transform:uppercase;color:#334155;padding:8px 12px;text-align:left">Section</th>
            <th style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.1em;
                       text-transform:uppercase;color:#334155;padding:8px 12px;text-align:left">Description</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)


# ── Page: Research ─────────────────────────────────────────────────────────────
elif page == "Research":
    st.markdown('<div class="hero-title" style="font-size:1.6rem">Research</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:1.5rem">Configure and run the multi-agent pipeline</div>', unsafe_allow_html=True)

    if not backend_ok:
        st.markdown("""
        <div class="metric-card" style="border-color:rgba(239,68,68,0.3);margin-bottom:1.5rem">
          <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#ef4444;">
            Backend offline. Start FastAPI: <code>uvicorn backend.main:app --reload</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.markdown("""<div class="section-header"><h2>Query</h2><span class="section-tag">Input</span></div>""", unsafe_allow_html=True)
        query = st.text_area(
            "Research query",
            placeholder="e.g. Analyze impact of LLMs on Indian fintech startups with data and trends",
            height=100,
            label_visibility="collapsed",
        )

    with c2:
        st.markdown("""<div class="section-header"><h2>Config</h2><span class="section-tag">Params</span></div>""", unsafe_allow_html=True)
        depth = st.selectbox("Research depth", ["quick", "detailed"], label_visibility="visible")
        domain = st.selectbox("Domain", ["general", "finance", "tech", "healthcare"], label_visibility="visible")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Pipeline", use_container_width=False, disabled=not backend_ok)

    if run_btn and query.strip():
        # agent stage stepper displayed while request is in flight
        agent_stages = [
            "Planner",
            "Researcher",
            "Verifier",
            "Writer",
            "Critic",
            "Evaluator",
            "Memory",
        ]

        st.markdown("""
        <div class="section-header" style="margin-top:1.5rem">
          <h2>Agent Progress</h2>
          <span class="section-tag">Live</span>
        </div>
        """, unsafe_allow_html=True)

        step_slots = []
        for stage in agent_stages:
            slot = st.empty()
            slot.markdown(f"""
            <div class="agent-step">
              <div class="agent-step-dot" style="background:#1e2330"></div>
              <span class="agent-step-label" style="color:#334155">{stage}</span>
              <span class="agent-step-status">waiting</span>
            </div>""", unsafe_allow_html=True)
            step_slots.append(slot)

        progress_bar = st.progress(0, text="Initialising pipeline...")

        def _update_step(idx: int, state: str):
            # updates a single agent step slot with state: running / done
            if state == "running":
                dot_color = "#f59e0b"
                label_color = "#e2e8f0"
                status_text = "running..."
            else:
                dot_color = "#10b981"
                label_color = "#64748b"
                status_text = "done"
            step_slots[idx].markdown(f"""
            <div class="agent-step">
              <div class="agent-step-dot" style="background:{dot_color}"></div>
              <span class="agent-step-label" style="color:{label_color}">{agent_stages[idx]}</span>
              <span class="agent-step-status">{status_text}</span>
            </div>""", unsafe_allow_html=True)

        # animate steps while the real API call runs in the background via a thread
        import threading

        result_container = {"data": None, "error": None}

        def _call_api():
            try:
                result_container["data"] = _run_research(query.strip(), depth, domain)
            except Exception as exc:
                result_container["error"] = str(exc)

        thread = threading.Thread(target=_call_api, daemon=True)
        thread.start()

        # step timing estimates (seconds): planner is fast, researcher is slow
        step_durations = [2, 0, 3, 3, 2, 2, 1]
        total_estimated = sum(step_durations) or 1

        elapsed = 0
        for i, duration in enumerate(step_durations):
            _update_step(i, "running")
            progress_bar.progress(
                min(int(elapsed / total_estimated * 85), 85),
                text=f"Running {agent_stages[i]}...",
            )
            # wait for either this step's budget or the thread to finish
            waited = 0
            while waited < duration and thread.is_alive():
                time.sleep(0.4)
                waited += 0.4
            _update_step(i, "done")
            elapsed += duration

        # wait for thread to actually finish
        thread.join(timeout=280)
        progress_bar.progress(100, text="Pipeline complete.")

        if result_container["error"]:
            st.error(f"Pipeline error: {result_container['error']}")
        else:
            st.session_state.research_result = result_container["data"]
            st.session_state.research_done = True
            st.session_state.last_query = query.strip()
            st.success("Research complete. View results in Sources and Evaluation tabs.")

    elif run_btn and not query.strip():
        st.warning("Enter a research query to continue.")

    # show report if already available
    if st.session_state.research_done and st.session_state.research_result:
        result = st.session_state.research_result
        st.markdown("""
        <div class="section-header" style="margin-top:2rem">
          <h2>Report</h2>
          <span class="section-tag">Output</span>
        </div>
        """, unsafe_allow_html=True)

        errors = result.get("errors", [])
        if errors:
            with st.expander("Pipeline Errors"):
                for e in errors:
                    st.markdown(f'<div class="mono-block" style="color:#ef4444">{e}</div>', unsafe_allow_html=True)

        report_text = result.get("report", "No report generated.")
        # st.markdown renders ## headings and * bullets; .report-md class styles them
        st.markdown('<div class="report-md">', unsafe_allow_html=True)
        st.markdown(report_text)
        st.markdown('</div>', unsafe_allow_html=True)

        # PDF export
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_bytes = _build_pdf(result)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
            file_name="research_report.pdf",
            mime="application/pdf",
        )


# ── Page: Sources ──────────────────────────────────────────────────────────────
elif page == "Sources":
    st.markdown('<div class="hero-title" style="font-size:1.6rem">Sources</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:1.5rem">Verified sources with credibility scoring</div>', unsafe_allow_html=True)

    if not st.session_state.research_done:
        st.markdown("""
        <div class="metric-card">
          <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748b;">
            No research data yet. Run the pipeline from the Research page.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        result = st.session_state.research_result
        sources = result.get("sources", [])

        # summary metrics
        if sources:
            avg_cred = sum(s.get("credibility_score", 0) for s in sources) / len(sources)
            high_cred = sum(1 for s in sources if s.get("credibility_score", 0) >= 0.7)
            m1, m2, m3 = st.columns(3)
            for col, label, val, sub in [
                (m1, "Total Sources",      str(len(sources)),        "verified"),
                (m2, "Avg Credibility",    f"{avg_cred:.2f}",        "0.0 – 1.0 scale"),
                (m3, "High Credibility",   str(high_cred),           "score >= 0.70"),
            ]:
                with col:
                    st.markdown(f"""<div class="metric-card">
                      <div class="metric-label">{label}</div>
                      <div class="metric-value">{val}</div>
                      <div class="metric-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="section-header" style="margin-top:2rem">
              <h2>Source List</h2>
              <span class="section-tag">Verified</span>
            </div>""", unsafe_allow_html=True)

            for s in sources:
                title    = s.get("title", "Untitled") or "Untitled"
                url      = s.get("url", "")
                snippet  = s.get("snippet", "")
                score    = s.get("credibility_score", 0)
                pill_html = _credibility_pill(score)
                st.markdown(f"""
                <div class="source-row">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.35rem">
                    <div class="source-title">{title}</div>
                    {pill_html}
                  </div>
                  <div class="source-url">{url}</div>
                  <div class="source-snippet">{snippet}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
              <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748b;">
                No verified sources in the last result.
              </div>
            </div>""", unsafe_allow_html=True)


# ── Page: Evaluation ───────────────────────────────────────────────────────────
elif page == "Evaluation":
    st.markdown('<div class="hero-title" style="font-size:1.6rem">Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:1.5rem">Pipeline quality metrics for the last research run</div>', unsafe_allow_html=True)

    if not st.session_state.research_done:
        st.markdown("""
        <div class="metric-card">
          <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748b;">
            No evaluation data yet. Run the pipeline from the Research page.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        result     = st.session_state.research_result
        ev         = result.get("evaluation", {})   # evaluator.py keys
        cr         = result.get("critique", {})     # critic.py keys

        # evaluator.py: factual_score is 0-10 integer
        factual       = ev.get("factual_score", 0)
        citation      = ev.get("citation_score", 0)
        hallucination = ev.get("hallucination_score", 0)
        latency       = ev.get("latency", 0)

        # normalise factual 0-10 -> 0.0-1.0 for colour coding
        factual_norm = factual / 10.0 if factual > 1.0 else float(factual)

        m1, m2, m3, m4 = st.columns(4)
        cards = [
            (m1, "Factual Score",       f"{factual}/10",          "LLM accuracy rating",  _score_color(factual_norm)),
            (m2, "Citation Score",      f"{citation:.2f}",        "0.0 – 1.0",            _score_color(citation)),
            (m3, "Hallucination Score", f"{hallucination:.2f}",   "lower is better",      _score_color(hallucination, invert=True)),
            (m4, "Latency",             f"{latency:.1f}s",        "pipeline time",        "var(--accent)"),
        ]
        for col, label, val, sub, color in cards:
            with col:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="font-size:1.6rem;color:{color}">{val}</div>
                  <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab_eval, tab_critic, tab_raw = st.tabs(["Evaluation Detail", "Critic Report", "Raw JSON"])

        with tab_eval:
            st.markdown("""<div class="section-header"><h2>Score Breakdown</h2><span class="section-tag">Metrics</span></div>""", unsafe_allow_html=True)
            # factual_reasoning is the one-sentence LLM explanation from evaluator.py
            factual_reasoning = ev.get("factual_reasoning", "")
            hallucinated_claims = ev.get("hallucinated_claims", [])

            lines = []
            lines.append(f"<b style='color:#e2e8f0'>factual_score</b>: {factual}/10")
            if factual_reasoning:
                lines.append(f"<b style='color:#e2e8f0'>factual_reasoning</b>: {factual_reasoning}")
            lines.append(f"<b style='color:#e2e8f0'>citation_score</b>: {citation}")
            lines.append(f"<b style='color:#e2e8f0'>hallucination_score</b>: {hallucination}")
            if hallucinated_claims:
                for claim in hallucinated_claims:
                    lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#ef4444'>&#9656; {claim}</span>")
            lines.append(f"<b style='color:#e2e8f0'>latency</b>: {latency}s")
            st.markdown(f'<div class="mono-block">{"<br>".join(lines)}</div>', unsafe_allow_html=True)

        with tab_critic:
            st.markdown("""<div class="section-header"><h2>Critic Analysis</h2><span class="section-tag">Quality</span></div>""", unsafe_allow_html=True)

            if not cr:
                st.markdown('<div class="mono-block" style="color:#64748b">No critic data available.</div>', unsafe_allow_html=True)
            else:
                # critic.py keys: quality_score, verdict, missing_insights, weak_citations, logical_gaps, suggestions
                quality_score  = cr.get("quality_score", "N/A")
                verdict        = cr.get("verdict", "N/A")
                missing        = cr.get("missing_insights", [])
                weak_citations = cr.get("weak_citations", [])
                logical_gaps   = cr.get("logical_gaps", [])
                suggestions    = cr.get("suggestions", [])

                verdict_pill = "pill-success" if str(verdict).lower() == "acceptable" else "pill-warn"
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:1.5rem">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <div class="metric-label">Quality Score</div>
                      <div class="metric-value" style="font-size:1.6rem">{quality_score}<span style="font-size:1rem;color:var(--muted)">/10</span></div>
                    </div>
                    <span class="pill {verdict_pill}" style="font-size:0.7rem;padding:6px 14px">{verdict}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                for col, label, items, tag_color in [
                    (c1, "Missing Insights", missing,        "#f59e0b"),
                    (c1, "Logical Gaps",     logical_gaps,   "#ef4444"),
                    (c2, "Weak Citations",   weak_citations, "#7c3aed"),
                    (c2, "Suggestions",      suggestions,    "#10b981"),
                ]:
                    with col:
                        if items:
                            items_html = "".join(
                                f'<li style="margin-bottom:6px;color:#94a3b8;font-family:Inter,sans-serif;font-size:0.78rem">{item}</li>'
                                for item in items
                            )
                            st.markdown(f"""
                            <div class="metric-card" style="margin-bottom:1rem">
                              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                          letter-spacing:0.1em;text-transform:uppercase;
                                          color:{tag_color};margin-bottom:0.6rem">{label}</div>
                              <ul style="padding-left:1rem;margin:0">{items_html}</ul>
                            </div>""", unsafe_allow_html=True)

        with tab_raw:
            st.markdown("""<div class="section-header"><h2>Raw JSON</h2><span class="section-tag">Debug</span></div>""", unsafe_allow_html=True)
            import json
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#00e5ff;margin-bottom:0.5rem">evaluation</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="mono-block">{json.dumps(ev, indent=2)}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#7c3aed;margin-bottom:0.5rem">critique</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="mono-block">{json.dumps(cr, indent=2)}</div>', unsafe_allow_html=True)


# ── Page: Follow-up Q&A ────────────────────────────────────────────────────────
elif page == "Follow-up Q&A":
    st.markdown('<div class="hero-title" style="font-size:1.6rem">Follow-up Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:1.5rem">Ask follow-up questions answered from LONG_TERM memory</div>', unsafe_allow_html=True)

    if not st.session_state.research_done:
        st.markdown("""
        <div class="metric-card">
          <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748b;">
            Run a research query first to populate memory.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        result = st.session_state.research_result

        # suggested follow-ups
        suggested = result.get("suggested_followups", [])
        if suggested:
            st.markdown("""
            <div class="section-header">
              <h2>Suggested Questions</h2>
              <span class="section-tag">From Critic</span>
            </div>""", unsafe_allow_html=True)
            for sq in suggested:
                st.markdown(f"""
                <div class="metric-card" style="padding:0.8rem 1.2rem;margin-bottom:0.5rem">
                  <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#94a3b8">&#9670; {sq}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header" style="margin-top:1.5rem">
          <h2>Ask a Question</h2>
          <span class="section-tag">Memory</span>
        </div>""", unsafe_allow_html=True)

        question = st.text_area(
            "Follow-up question",
            placeholder="e.g. What are the main risks mentioned for fintech startups?",
            height=80,
            label_visibility="collapsed",
        )

        ask_btn = st.button("Ask", use_container_width=False, disabled=not backend_ok)

        if ask_btn and question.strip():
            with st.spinner("Querying memory..."):
                try:
                    resp = _run_followup(question.strip(), st.session_state.last_query)
                    answer = resp.get("answer", "No answer returned.")
                    followup_errors = resp.get("errors", [])
                except Exception as exc:
                    answer = None
                    followup_errors = [str(exc)]

            if answer:
                st.markdown("""
                <div class="section-header">
                  <h2>Answer</h2>
                  <span class="section-tag">Output</span>
                </div>""", unsafe_allow_html=True)
                st.markdown(f'<div class="report-block">{answer}</div>', unsafe_allow_html=True)

            if followup_errors:
                for e in followup_errors:
                    st.markdown(f'<div class="mono-block" style="color:#ef4444;margin-top:0.5rem">{e}</div>', unsafe_allow_html=True)

        elif ask_btn and not question.strip():
            st.warning("Enter a question to continue.")