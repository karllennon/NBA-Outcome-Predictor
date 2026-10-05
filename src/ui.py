"""
Visual building blocks for the dashboard: CSS, cards, team badges, pills, and a Plotly theme.

Colors: model = blue (#3987e5) and market = orange (#d95926) are categorical slots 1-2 of the
dark reference palette; status colors are reserved for good / warning / critical states and
always come with a text label. All clear 3:1 contrast on the navy surfaces used here.
"""
import html
import streamlit as st

INK = '#e6ebf5'
INK_2 = '#93a0b8'
INK_3 = '#6b7a94'
SURFACE = '#131c2e'
MODEL = '#3987e5'
MARKET = '#d95926'
HOME_SIDE = '#3987e5'   # diverging pair for "favors home / favors away"
AWAY_SIDE = '#e66767'
GOOD, WARN, CRIT = '#0ca30c', '#fab219', '#d03b3b'

CSS = """
<style>
.stApp {
  background: radial-gradient(1100px 520px at 0% -10%, #1a2a49 0%, rgba(11,18,32,0) 60%),
              radial-gradient(900px 500px at 100% 0%, #141f37 0%, rgba(11,18,32,0) 55%), #0b1220;
}
[data-testid="stHeader"] { background: rgba(11,18,32,0.85); backdrop-filter: blur(6px); }
.block-container { padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1560px; }
h1, h2, h3 { letter-spacing: -0.01em; }

/* Cards: pure-HTML cards and Streamlit containers whose key starts with "card" */
.cv-card, div[class*="st-key-card"] {
  background: linear-gradient(180deg, rgba(25,36,59,0.94), rgba(17,26,43,0.94));
  border: 1px solid rgba(132,160,210,0.16);
  border-radius: 14px;
  padding: 14px 16px 12px 16px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.03);
  margin-bottom: 14px;
}
div[class*="st-key-card"] { gap: 0.55rem; }
.cv-title {
  font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: #8fa0bd; font-weight: 700; margin: 0 0 10px 0;
  display: flex; justify-content: space-between; align-items: center;
}
.cv-title .cv-sub { letter-spacing: 0.02em; text-transform: none; font-weight: 500; color: #6b7a94; }
.cv-muted { color: #93a0b8; font-size: 0.82rem; }
.cv-small { font-size: 0.78rem; color: #93a0b8; }

/* Page header */
.cv-header { display:flex; align-items:flex-end; justify-content:space-between; margin: 0 0 14px 0; }
.cv-brand { font-size: 0.78rem; letter-spacing: 0.16em; text-transform: uppercase; color:#8fa0bd; font-weight:700; }
.cv-date { font-size: 1.9rem; font-weight: 800; color: #f2f5fb; line-height: 1.1; }
.cv-mode { font-size: 0.8rem; color:#93a0b8; }

/* Team badges */
.cv-badge {
  display:inline-flex; align-items:center; justify-content:center;
  min-width: 46px; height: 28px; padding: 0 8px; border-radius: 8px;
  font-weight: 800; font-size: 0.8rem; letter-spacing: 0.04em;
  border: 1px solid rgba(255,255,255,0.22); box-shadow: inset 0 -2px 0 rgba(0,0,0,0.25);
}
.cv-badge.lg { min-width: 74px; height: 46px; font-size: 1.15rem; border-radius: 12px; }
.cv-badge.sm { min-width: 38px; height: 22px; font-size: 0.7rem; padding: 0 5px; border-radius: 6px; }
td.cv-game { white-space: nowrap; }

/* Pills (status always has a text label) */
.cv-pill { display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
           letter-spacing: 0.03em; border: 1px solid transparent; white-space: nowrap; }
.cv-pill.good { background: rgba(12,163,12,0.16); color: #63d463; border-color: rgba(12,163,12,0.45); }
.cv-pill.warn { background: rgba(250,178,25,0.14); color: #fab219; border-color: rgba(250,178,25,0.45); }
.cv-pill.crit { background: rgba(208,59,59,0.16); color: #ff8a8a; border-color: rgba(208,59,59,0.5); }
.cv-pill.neutral { background: rgba(147,160,184,0.12); color: #b6c1d6; border-color: rgba(147,160,184,0.3); }
.cv-pill.model { background: rgba(57,135,229,0.16); color: #8fbcf3; border-color: rgba(57,135,229,0.45); }
.cv-pill.market { background: rgba(217,89,38,0.16); color: #f19a75; border-color: rgba(217,89,38,0.5); }

/* Matchup */
.cv-matchup { display:grid; grid-template-columns: 1fr auto 1fr; align-items:center; gap: 10px; }
.cv-team { display:flex; flex-direction:column; align-items:center; gap:6px; text-align:center; }
.cv-team-name { font-weight: 700; color:#f2f5fb; font-size: 1.02rem; }
.cv-team-rec { color:#93a0b8; font-size: 0.78rem; }
.cv-vs { color:#6b7a94; font-weight:800; font-size: 0.9rem; text-align:center; }
.cv-score { font-size: 1.7rem; font-weight: 800; color:#f2f5fb; }
.cv-score.win { color:#63d463; }
.cv-meta { text-align:center; color:#93a0b8; font-size:0.8rem; margin-top: 8px; }

/* Probability split bar */
.cv-split { display:flex; height: 12px; border-radius: 999px; overflow:hidden; gap: 2px; margin: 8px 0 4px 0; }
.cv-split > div { height:100%; }
.cv-bignum { font-size: 2.1rem; font-weight: 800; color:#f2f5fb; line-height: 1; }
.cv-prob-grid { display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
.cv-prob-cell { background: rgba(10,16,29,0.55); border:1px solid rgba(132,160,210,0.12); border-radius: 10px; padding: 10px 12px; }
.cv-prob-label { font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#8fa0bd; font-weight:700; }
.cv-prob-val { font-size: 1.45rem; font-weight: 800; color:#f2f5fb; margin-top: 2px; }
.cv-prob-val .sub { font-size: 0.78rem; color:#93a0b8; font-weight:600; margin-left: 4px; }

/* Tables */
table.cv-table { width:100%; border-collapse: collapse; font-size: 0.84rem; }
table.cv-table th { text-align:left; color:#8fa0bd; font-weight:700; font-size:0.66rem; letter-spacing:0.08em;
                    text-transform:uppercase; padding: 6px 4px; border-bottom: 1px solid rgba(132,160,210,0.14); }
table.cv-table td { padding: 7px 4px; border-bottom: 1px solid rgba(132,160,210,0.08); color:#dfe5f1; vertical-align: middle; }
table.cv-table tr.hl td { background: rgba(250,178,25,0.07); }
table.cv-table tr.sel td { background: rgba(57,135,229,0.10); }
table.cv-table td.num { text-align:right; font-variant-numeric: tabular-nums; }
table.cv-table th.num { text-align:right; }

/* Team form comparison */
.cv-form-row { display:grid; grid-template-columns: 54px 1fr 1fr 54px; align-items:center; gap: 6px; padding: 1px 0 6px 0; }
.cv-form-head { display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px; }
.cv-form-val { font-variant-numeric: tabular-nums; font-weight: 700; color:#dfe5f1; font-size: 0.84rem; }
.cv-form-val.better { color:#f2f5fb; }
.cv-form-label { text-align:center; color:#93a0b8; font-size: 0.72rem; margin-top: 2px; }
.cv-track { height: 6px; background: rgba(147,160,184,0.12); border-radius: 999px; overflow:hidden; display:flex; }
.cv-track.left { justify-content:flex-end; }
.cv-track > div { height:100%; border-radius: 999px; }

/* Injury chips */
.cv-inj { display:flex; justify-content:space-between; gap: 8px; padding: 6px 0; border-bottom: 1px solid rgba(132,160,210,0.08); }
.cv-inj:last-child { border-bottom: none; }
.cv-inj-name { color:#eef2f9; font-weight: 600; font-size: 0.84rem; }
.cv-inj-reason { color:#93a0b8; font-size: 0.74rem; text-align:right; }

/* Game list buttons */
div[class*="st-key-game-"] button {
  width: 100%; justify-content: flex-start; text-align: left;
  background: rgba(10,16,29,0.6); border: 1px solid rgba(132,160,210,0.14); border-radius: 10px;
  padding: 0.55rem 0.7rem; color: #dfe5f1; font-variant-numeric: tabular-nums;
}
div[class*="st-key-game-"] button:hover { border-color: rgba(57,135,229,0.65); color: #ffffff; }
div[class*="st-key-game-"] button[kind="primary"] {
  background: linear-gradient(90deg, rgba(57,135,229,0.28), rgba(57,135,229,0.10));
  border-color: rgba(57,135,229,0.85); color: #ffffff;
}
div[class*="st-key-game-"] button p { font-size: 0.86rem; }

/* Metric tiles */
.cv-tiles { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }
.cv-tile { background: linear-gradient(180deg, rgba(25,36,59,0.94), rgba(17,26,43,0.94));
           border: 1px solid rgba(132,160,210,0.16); border-radius: 14px; padding: 12px 14px; }
.cv-tile .k { font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; color:#8fa0bd; font-weight:700; }
.cv-tile .v { font-size: 1.6rem; font-weight: 800; color:#f2f5fb; margin-top: 4px; }
.cv-tile .s { font-size: 0.78rem; color:#93a0b8; }
@media (max-width: 900px) { .cv-tiles { grid-template-columns: repeat(2, 1fr); } }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text):
    return html.escape(str(text))


def badge(abbr, size='', colors=None):
    from dashboard_data import TEAM_COLORS, LIGHT_BADGES
    colors = colors or TEAM_COLORS
    bg = colors.get(abbr, '#33415c')
    fg = '#0b1220' if abbr in LIGHT_BADGES else '#ffffff'
    return f'<span class="cv-badge {size}" style="background:{bg};color:{fg}">{esc(abbr)}</span>'


def pill(text, kind='neutral'):
    return f'<span class="cv-pill {kind}">{esc(text)}</span>'


def card(title, body_html, sub=None):
    sub_html = f'<span class="cv-sub">{esc(sub)}</span>' if sub else ''
    return f'<div class="cv-card"><div class="cv-title"><span>{esc(title)}</span>{sub_html}</div>{body_html}</div>'


def card_title(title, sub=None):
    sub_html = f'<span class="cv-sub">{esc(sub)}</span>' if sub else ''
    st.markdown(f'<div class="cv-title"><span>{esc(title)}</span>{sub_html}</div>', unsafe_allow_html=True)


def html_block(body):
    st.markdown(body, unsafe_allow_html=True)


def pct(p, digits=0):
    return '–' if p is None or p != p else f"{p * 100:.{digits}f}%"


def plotly_layout(fig, height=260, legend=True, margin=None):
    """Recessive axes and grid, transparent background, ink-colored text."""
    fig.update_layout(
        height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=INK_2, size=12),
        margin=margin or dict(l=8, r=12, t=8, b=8),
        hoverlabel=dict(bgcolor='#0e1626', bordercolor='rgba(132,160,210,0.35)', font=dict(color=INK, size=12)),
        showlegend=legend,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                    font=dict(color=INK_2, size=11), bgcolor='rgba(0,0,0,0)'),
    )
    fig.update_xaxes(gridcolor='rgba(147,160,184,0.08)', zeroline=False, linecolor='rgba(147,160,184,0.25)',
                     tickfont=dict(color=INK_3))
    fig.update_yaxes(gridcolor='rgba(147,160,184,0.08)', zeroline=False, linecolor='rgba(147,160,184,0.25)',
                     tickfont=dict(color=INK_3))
    return fig


PLOTLY_CONFIG = {'displayModeBar': False, 'responsive': True}
