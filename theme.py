"""
theme.py (V3)
Design system for Telecom Churn Intelligence (TCI).

V2's theme.py hardcoded a fixed dark navy palette in every custom card. That
broke the moment a user switched Streamlit's native Light/Dark/System theme
toggle (Settings menu, top-right) to Light — the cards would still be
dark-hardcoded, recreating the exact same class of contrast bug in reverse.

V3 fixes this at the root: every card/box below uses Streamlit's own CSS
variables (--background-color, --text-color, --secondary-background-color,
--primary-color), which Streamlit itself keeps correct under Light, Dark,
and System modes automatically. Risk colors (red/amber/green) are the one
deliberate exception — risk semantics must stay visually consistent
regardless of theme, so those stay fixed hex values.
"""

import streamlit as st

RISK_COLORS = {"Low": "#16A34A", "Medium": "#D97706", "High": "#DC2626"}
RISK_BG = {
    "Low": "rgba(22,163,74,0.12)",
    "Medium": "rgba(217,119,6,0.12)",
    "High": "rgba(220,38,38,0.12)",
}

BRAND_PRIMARY = "#0F4C81"   # deep telecom blue
BRAND_ACCENT = "#22D3EE"    # cyan accent


def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 1200px; }

        .app-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            color: var(--text-color);
        }

        .kpi-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 10px;
            padding: 1.05rem 1rem;
            text-align: center;
        }
        .kpi-value { font-size: 1.8rem; font-weight: 700; color: var(--text-color); line-height: 1.2; }
        .kpi-label { font-size: 0.85rem; color: var(--text-color); opacity: 0.65; margin-top: 0.3rem; }
        .kpi-icon { font-size: 1.25rem; margin-bottom: 0.15rem; }

        .section-title { font-size: 1.25rem; font-weight: 700; color: var(--text-color); margin: 0.3rem 0 0.4rem 0; }
        .section-caption { color: var(--text-color); opacity: 0.65; font-size: 0.92rem; margin-bottom: 0.8rem; }

        .risk-badge { display: inline-block; padding: 0.32rem 0.85rem; border-radius: 999px; font-weight: 700; font-size: 0.92rem; }
        .risk-badge-low { background-color: """ + RISK_BG["Low"] + """; color: """ + RISK_COLORS["Low"] + """; }
        .risk-badge-medium { background-color: """ + RISK_BG["Medium"] + """; color: """ + RISK_COLORS["Medium"] + """; }
        .risk-badge-high { background-color: """ + RISK_BG["High"] + """; color: """ + RISK_COLORS["High"] + """; }

        .capability-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.25);
            border-left: 4px solid """ + BRAND_PRIMARY + """;
            border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem; color: var(--text-color);
        }
        .capability-title { font-weight: 700; margin-bottom: 0.2rem; }

        .driver-row {
            display: flex; justify-content: space-between; align-items: center;
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.25); border-left: 4px solid """ + BRAND_ACCENT + """;
            border-radius: 8px; padding: 0.55rem 0.9rem; margin-bottom: 0.45rem; color: var(--text-color);
        }

        .hero-title { font-size: 2.1rem; font-weight: 800; color: var(--text-color); margin-bottom: 0.1rem; }
        .hero-tagline { font-size: 1.1rem; color: """ + BRAND_PRIMARY + """; font-weight: 600; margin-bottom: 0.7rem; }

        .disclaimer-box {
            background-color: rgba(15,76,129,0.08); border: 1px solid """ + BRAND_PRIMARY + """;
            border-radius: 8px; padding: 0.75rem 1.05rem; color: var(--text-color); font-size: 0.88rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon: str, value: str, label: str):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


RISK_EMOJI = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}


def risk_badge_html(risk_level: str) -> str:
    cls = {"Low": "risk-badge-low", "Medium": "risk-badge-medium", "High": "risk-badge-high"}[risk_level]
    return f'<span class="risk-badge {cls}">{RISK_EMOJI[risk_level]} {risk_level} Risk</span>'


def section_header(title: str, caption: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def app_card_start():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)


def app_card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def capability_card(title: str, description: str):
    st.markdown(
        f'<div class="capability-card"><div class="capability-title">{title}</div>{description}</div>',
        unsafe_allow_html=True,
    )


def disclaimer(text: str):
    st.markdown(f'<div class="disclaimer-box">ℹ️ {text}</div>', unsafe_allow_html=True)
