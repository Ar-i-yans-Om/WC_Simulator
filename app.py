"""
FIFA WC 2026 SIMULATION ENGINE — Dashboard UI
=============================================
Streamlit dashboard. Reads data via ui_data.py (fixtures + results + predictions).
The UI is a thin render layer: it never calls the pipeline, only reads JSON the
pipeline writes. See ui_data.py for the predictions.json contract.

Run:  streamlit run app.py
"""

import re

import streamlit as st

import ui_data as D

# --------------------------------------------------------------------------- #
#  PAGE CONFIG
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="FIFA WC 2026 Simulation Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
#  THEME PALETTE
# --------------------------------------------------------------------------- #
PRIMARY = "#0a1f44"
BLUE = "#1a5fd4"
BLUE_SOFT = "#e8f0fe"
WIN = "#1a5fd4"
DRAW = "#cbd5e1"
LOSS = "#0a1f44"
INK = "#0a1f44"
MUTED = "#64748b"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Barlow+Condensed:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}
    .stApp {{ background: #ffffff; }}
    .block-container {{ padding-top: 1.4rem; max-width: 1180px; }}

    /* ---------- HEADER ---------- */
    .app-header {{
        display: flex; align-items: center; gap: 14px;
        padding: 18px 26px; margin-bottom: 18px;
        background: linear-gradient(100deg, {PRIMARY} 0%, {BLUE} 130%);
        border-radius: 14px; box-shadow: 0 6px 22px rgba(10,31,68,0.18);
    }}
    .app-header .ball {{ font-size: 34px; line-height: 1; }}
    .app-header h1 {{
        font-family: 'Barlow Condensed', sans-serif; color: #fff; margin: 0;
        font-size: 36px; font-weight: 800; letter-spacing: 1.5px;
    }}
    .app-header .sub {{
        color: #b9cdf2; font-size: 12.5px; font-weight: 500;
        letter-spacing: 2px; text-transform: uppercase; margin-top: 2px;
    }}

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid #eef2f7; }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600; font-size: 14px; color: {MUTED};
        padding: 8px 16px; border-radius: 8px 8px 0 0;
    }}
    .stTabs [aria-selected="true"] {{ color: {BLUE} !important; background: {BLUE_SOFT}; }}
    .stTabs [aria-disabled="true"] {{ color: #cbd5e1 !important; }}

    /* ---------- SEGMENTED CONTROL (LIVE | PROJECTED), centered ---------- */
    div[data-testid="stSegmentedControl"] {{
        width: 100% !important; display: flex !important;
        justify-content: center !important; align-items: center !important;
    }}
    div[data-testid="stSegmentedControl"] > * {{
        margin-left: auto !important; margin-right: auto !important;
        display: inline-flex !important; width: fit-content !important;
        background: #eef1f5 !important; border-radius: 10px; padding: 3px;
        gap: 3px; border: 1px solid #e2e8f0 !important;
    }}
    /* base segment: white, grey text, no red accent anywhere */
    div[data-testid="stSegmentedControl"] button {{
        border: 1px solid transparent !important;
        background: #ffffff !important;
        color: {INK} !important;
        font-weight: 700 !important; font-size: 12px !important;
        letter-spacing: 1.5px !important;
        padding: 6px 22px !important; border-radius: 8px !important;
        box-shadow: none !important; outline: none !important;
    }}
    div[data-testid="stSegmentedControl"] button p {{ color: {INK} !important; }}
    /* hover / focus must not reintroduce the theme's red */
    div[data-testid="stSegmentedControl"] button:hover,
    div[data-testid="stSegmentedControl"] button:focus,
    div[data-testid="stSegmentedControl"] button:active {{
        color: {INK} !important; border-color: #e2e8f0 !important;
        box-shadow: none !important; outline: none !important;
    }}
    /* selected segment: very light grey, dark text (override blue theme accent) */
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-selected="true"],
    div[data-testid="stSegmentedControl"] button[data-selected="true"],
    div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {{
        background: #e2e8f0 !important;
        color: {INK} !important;
        border-color: #cbd5e1 !important;
        box-shadow: inset 0 1px 2px rgba(10,31,68,0.06) !important;
    }}
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] *,
    div[data-testid="stSegmentedControl"] button[aria-selected="true"] *,
    div[data-testid="stSegmentedControl"] button[data-selected="true"] *,
    div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] * {{
        color: {INK} !important;
    }}

    /* ---------- STANDINGS TABLE ---------- */
    .tbl-card {{
        border: 1px solid #eef2f7; border-radius: 12px;
        padding: 14px 16px 10px 16px; background: #fff;
        box-shadow: 0 2px 10px rgba(10,31,68,0.05); transition: all .25s ease;
    }}
    .tbl-card.dim {{ opacity: 0.40; filter: grayscale(0.4); }}
    .tbl-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
    .tbl-head .lbl {{ font-size:12px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:{BLUE}; }}
    .tbl-head .lbl.pred {{ color:{PRIMARY}; }}
    .badge {{ font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px;
        background:{BLUE_SOFT}; color:{BLUE}; letter-spacing:1px; }}
    table.stand {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
    table.stand th {{ text-align:left; color:{MUTED}; font-weight:600; font-size:11px;
        text-transform:uppercase; letter-spacing:0.5px; padding:4px 6px; border-bottom:1px solid #eef2f7; }}
    table.stand th.num, table.stand td.num {{ text-align:center; }}
    table.stand td {{ padding:7px 6px; border-bottom:1px solid #f5f7fa; color:{INK}; }}
    table.stand tr:last-child td {{ border-bottom:none; }}
    table.stand td.team {{ font-weight:600; }}
    table.stand td.pts {{ font-weight:800; color:{BLUE}; }}
    .pos {{ display:inline-block; width:20px; height:20px; line-height:20px; text-align:center;
        border-radius:6px; font-size:11px; font-weight:700; background:#f1f5f9; color:{MUTED}; margin-right:2px; }}
    .pos.q {{ background:{BLUE}; color:#fff; }}
    .delta-up {{ color:#16a34a; font-weight:700; }}
    .delta-down {{ color:#dc2626; font-weight:700; }}
    .delta-flat {{ color:{MUTED}; }}

    /* ---------- MATCH PROB BAR ---------- */
    .match-teams {{ display:flex; justify-content:space-between; font-size:14px;
        font-weight:600; color:{INK}; margin-bottom:6px; }}
    .match-teams .vs {{ color:{MUTED}; font-weight:500; font-size:12px; }}
    .prob-bar {{ display:flex; width:100%; height:34px; border-radius:8px; overflow:hidden;
        box-shadow: inset 0 0 0 1px #eef2f7; }}
    .prob-seg {{ display:flex; align-items:center; justify-content:center; font-size:12.5px;
        font-weight:700; color:#fff; white-space:nowrap; transition:all .3s ease; }}
    .seg-win {{ background:{WIN}; }}
    .seg-draw {{ background:{DRAW}; color:{INK}; }}
    .seg-loss {{ background:{LOSS}; }}
    .prob-pending {{ display:flex; align-items:center; justify-content:center; width:100%;
        height:34px; border-radius:8px; background:#f8fafc; color:{MUTED};
        font-size:12px; font-weight:600; box-shadow: inset 0 0 0 1px #eef2f7; }}
    .prob-legend {{ display:flex; gap:18px; margin:4px 2px 0 2px; font-size:11px; color:{MUTED}; }}
    .prob-legend .k {{ display:flex; align-items:center; gap:5px; }}
    .dot {{ width:9px; height:9px; border-radius:3px; display:inline-block; }}

    /* ---------- DETAIL DROPDOWN ---------- */
    .detail-block .h {{ font-size:10.5px; font-weight:700; letter-spacing:1px; text-transform:uppercase;
        color:{MUTED}; margin-bottom:4px; }}
    .detail-block .v {{ font-size:18px; font-weight:800; color:{BLUE}; font-family:'Barlow Condensed',sans-serif; }}
    .scoreline-pill {{ display:inline-block; background:{BLUE_SOFT}; color:{BLUE}; font-weight:700;
        padding:3px 10px; border-radius:7px; margin:2px 6px 2px 0; font-size:12.5px; }}
    .proj-note {{ margin-top:10px; padding:7px 10px; border-radius:8px; background:#fff8e6;
        color:#92710c; font-size:11.5px; font-weight:600; border:1px solid #f3e2b3; }}

    /* ---------- EXPANDER ("Match details") ---------- */
    details[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary {{
        display: flex !important; align-items: center !important;
        justify-content: center !important; text-align: center !important;
        color: {INK} !important; font-weight: 600 !important;
    }}
    details[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span {{
        color: {INK} !important; font-weight: 600 !important;
        text-align: center !important; width: auto !important; flex: 0 0 auto !important;
    }}
    div[data-testid="stExpander"] summary svg {{ fill: {INK} !important; color: {INK} !important; }}
    div[data-testid="stExpander"] details {{ border-radius: 10px !important; }}
    .md-divider {{ display:flex; align-items:center; gap:10px; margin:18px 0 10px 0; }}
    .md-divider .lbl {{ font-family:'Barlow Condensed',sans-serif; font-size:14px; font-weight:700;
        letter-spacing:1.5px; text-transform:uppercase; color:{BLUE}; white-space:nowrap; }}
    .md-divider .line {{ flex:1; height:1px; background:#eef2f7; }}
    .md-divider .tag {{ font-size:10px; font-weight:600; color:{MUTED}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
#  HEADER
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="app-header">
        <span class="ball">⚽</span>
        <div>
            <h1>FIFA WC 2026 SIMULATION ENGINE</h1>
            <div class="sub">Multi-Agent Match Prediction · Group Stage</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
#  LOAD DATA (cached; invalidates when any source file's mtime changes)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _bundle(_fx_m, _rs_m, _pr_m):
    return D.load_fixtures(), D.load_results(), D.load_predictions()


def _mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


fixtures, results, predictions = _bundle(
    _mtime(D.FIXTURES_FILE), _mtime(D.RESULTS_FILE), _mtime(D.PREDICTIONS_FILE)
)

GROUPS = D.groups_in_order(fixtures)

# --------------------------------------------------------------------------- #
#  RENDER HELPERS
# --------------------------------------------------------------------------- #
def standings_html(rows, predicted=False):
    head = (
        "<tr><th>#</th><th>Team</th>"
        "<th class='num'>P</th><th class='num'>W</th><th class='num'>D</th>"
        "<th class='num'>L</th><th class='num'>GD</th><th class='num'>Pts</th>"
        + ("<th class='num'>±</th>" if predicted else "")
        + "</tr>"
    )
    body = ""
    for i, r in enumerate(rows, start=1):
        gd = r["GD"]
        gd_s = f"+{gd}" if gd > 0 else str(gd)
        pos_cls = "pos q" if i <= 2 else "pos"
        delta_cell = ""
        if predicted:
            mv = r.get("delta", "flat")
            sym = {"up": "▲", "down": "▼", "flat": "—"}[mv]
            cls = {"up": "delta-up", "down": "delta-down", "flat": "delta-flat"}[mv]
            delta_cell = f"<td class='num {cls}'>{sym}</td>"
        body += (
            f"<tr><td><span class='{pos_cls}'>{i}</span></td>"
            f"<td class='team'>{r['team']}</td>"
            f"<td class='num'>{r['P']}</td><td class='num'>{r['W']}</td>"
            f"<td class='num'>{r['D']}</td><td class='num'>{r['L']}</td>"
            f"<td class='num'>{gd_s}</td><td class='num pts'>{r['Pts']}</td>"
            f"{delta_cell}</tr>"
        )
    return f"<table class='stand'>{head}{body}</table>"


def standings_card(rows, kind, focused, status=None, show_header=True):
    label = "Current Table" if kind == "current" else "Predicted Table"
    lbl_cls = "lbl" if kind == "current" else "lbl pred"
    badge = "LIVE" if kind == "current" else "PROJECTED"
    dim = "" if focused else "dim"
    inner = standings_html(rows, predicted=(kind == "predicted"))

    note = ""
    if kind == "predicted" and status is not None:
        if status["total"] and status["predicted"] == 0 and status["played"] == 0:
            note = ("<div class='proj-note'>No predictions yet — run the pipeline "
                    "to project this group.</div>")
        elif not status["fully_projected"]:
            done = status["played"] + status["predicted"]
            note = (f"<div class='proj-note'>Partial projection · {done}/{status['total']} "
                    f"matches predicted. Remaining fixtures assume current standing.</div>")

    header = ""
    if show_header:
        header = (f"<div class='tbl-head'><span class='{lbl_cls}'>{label}</span>"
                  f"<span class='badge'>{badge}</span></div>")

    st.markdown(
        f"""
        <div class="tbl-card {dim}">
            {header}
            {inner}
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def match_block(m):
    a, b = m["home"], m["away"]
    pa, pd, pb = m["prob_home_win"], m["prob_draw"], m["prob_away_win"]
    have = m["has_prediction"] and None not in (pa, pd, pb)

    st.markdown(
        f"<div class='match-teams'><span>{a}</span>"
        f"<span class='vs'>vs</span><span>{b}</span></div>",
        unsafe_allow_html=True,
    )

    if have:
        st.markdown(
            f"""
            <div class="prob-bar">
                <div class="prob-seg seg-win"  style="width:{pa}%">{pa}%</div>
                <div class="prob-seg seg-draw" style="width:{pd}%">{pd}%</div>
                <div class="prob-seg seg-loss" style="width:{pb}%">{pb}%</div>
            </div>
            <div class="prob-legend">
                <span class="k"><span class="dot" style="background:{WIN}"></span>{a} win</span>
                <span class="k"><span class="dot" style="background:{DRAW}"></span>Draw</span>
                <span class="k"><span class="dot" style="background:{LOSS}"></span>{b} win</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='prob-pending'>Prediction pending — run the pipeline for this fixture</div>",
            unsafe_allow_html=True,
        )

    # ---- per-match detail dropdown (collapsed by default) ----
    with st.expander("Match details", expanded=False):
        if not m["has_prediction"]:
            st.caption("No prediction available for this fixture yet.")
        else:
            eg = m.get("expected_goals") or {}
            tops = m.get("top_scorelines") or []
            sl = m.get("predicted_scoreline")   # bare score like "2-0" from predictions.json

            cols = st.columns(3)
            with cols[0]:
                xg = "—"
                if eg.get("home") is not None and eg.get("away") is not None:
                    xg = f"{eg['home']:.1f} – {eg['away']:.1f}"
                st.markdown(
                    f"<div class='detail-block'><div class='h'>Expected Goals</div>"
                    f"<div class='v'>{xg}</div></div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                # extract just the "h-a" digits, so names are added exactly once
                # even if an older predictions.json baked names into the string.
                sl_display = "—"
                if sl:
                    _m = re.search(r"(\d+)\s*-\s*(\d+)", str(sl))
                    if _m:
                        sl_display = f"{a} {_m.group(1)}-{_m.group(2)} {b}"
                st.markdown(
                    f"<div class='detail-block'><div class='h'>Predicted Scoreline</div>"
                    f"<div class='v'>{sl_display}</div></div>",
                    unsafe_allow_html=True,
                )
            with cols[2]:
                conf = f"{max(pa, pd, pb)}%" if None not in (pa, pd, pb) else "—"
                st.markdown(
                    f"<div class='detail-block'><div class='h'>Top Outcome</div>"
                    f"<div class='v'>{conf}</div></div>",
                    unsafe_allow_html=True,
                )

            if tops:
                pills = "".join(
                    f"<span class='scoreline-pill'>{t.get('score')} · {t.get('prob')}%</span>"
                    for t in tops
                )
                st.markdown(
                    f"<div style='margin-top:12px'><div class='detail-block'>"
                    f"<div class='h'>Most Likely Scorelines</div>{pills}</div></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


def render_group(letter):
    cur = D.current_standings(fixtures, results, letter)
    pred = D.predicted_standings(fixtures, results, predictions, letter)
    matches = D.match_predictions(fixtures, predictions, letter)
    status = D.group_prediction_status(fixtures, results, predictions, letter)

    # focus state
    state_key = f"focus_state_{letter}"
    if state_key not in st.session_state:
        st.session_state[state_key] = "LIVE"
    focus = st.session_state[state_key]

    # --- centered LIVE | PROJECTED toggle ---
    tl, tc, tr = st.columns([2, 1, 2])
    with tc:
        sc = st.segmented_control(
            "focus", ["LIVE", "PROJECTED"],
            default=focus, key=f"seg_{letter}",
            label_visibility="collapsed",
        )
    if sc and sc != focus:
        st.session_state[state_key] = sc
        st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # --- the two tables (headers rendered inside the cards) ---
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        standings_card(cur, "current", focused=(focus == "LIVE"))
    with c2:
        standings_card(pred, "predicted", focused=(focus == "PROJECTED"),
                       status=status)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style=\"font-family:'Barlow Condensed',sans-serif;font-size:22px;"
        "font-weight:700;color:#0a1f44;letter-spacing:0.5px;text-transform:uppercase\">"
        "Match Probabilities</div>",
        unsafe_allow_html=True,
    )
    st.caption("Win / Draw / Loss projection per group fixture")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # group fixtures by matchday with a divider per MD
    by_md = status["by_md"]
    last_md = None
    for m in matches:
        md = m.get("md")
        if md != last_md:
            slot = by_md.get(md, {})
            done = slot.get("played", 0) + slot.get("predicted", 0)
            total = slot.get("total", 0)
            tag = f"{done}/{total} predicted" if total else ""
            st.markdown(
                f"<div class='md-divider'><span class='lbl'>Matchday {md}</span>"
                f"<span class='line'></span><span class='tag'>{tag}</span></div>",
                unsafe_allow_html=True,
            )
            last_md = md
        match_block(m)


# --------------------------------------------------------------------------- #
#  TOP-LEVEL TABS
# --------------------------------------------------------------------------- #
tab_groups, tab_bracket = st.tabs(["Groups", "Predicted Knockout Bracket"])

with tab_groups:
    if not GROUPS:
        st.warning("No fixtures found. Check that data/fixtures.json exists at the project root.")
    else:
        group_tabs = st.tabs([f"Group {g}" for g in GROUPS])
        for tab, g in zip(group_tabs, GROUPS):
            with tab:
                render_group(g)

with tab_bracket:
    st.info("Knockout bracket coming soon — disabled for now.")