import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(__file__))

import dashboard_data as dd
import ui
from inference import GamePredictor
from matchups import FEATURES
from train import feature_weights

st.set_page_config(page_title="NBA Outcome Predictor", page_icon="🏀", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()


# ─────────────────────────────────────────────────────────────── cached data

@st.cache_resource(ttl=900)
def load_predictor():
    predictor = GamePredictor()
    predictor.load_injury_report()  # re-downloaded at most every 15 minutes
    return predictor


@st.cache_resource
def load_model():
    return joblib.load('models/nba_model.joblib')


@st.cache_data
def load_games():
    return pd.read_csv('data/raw_nba_data.csv')


@st.cache_data
def load_test_predictions():
    return pd.read_csv('data/test_predictions.csv', parse_dates=['GAME_DATE'])


@st.cache_data
def load_metrics():
    m = pd.read_csv('data/test_metrics.csv')
    mm = pd.read_csv('data/market_metrics.csv') if os.path.exists('data/market_metrics.csv') else None
    return m, mm


@st.cache_data(ttl=600, show_spinner="Loading today's games, injury report and Kalshi prices...")
def live_view():
    """Today's games; building it also logs pre-tip-off predictions (at most every 10 min)."""
    return dd.day_view_live(load_predictor())


@st.cache_data(show_spinner=False)
def replay_view(date_str):
    return dd.day_view_replay(date_str)


@st.cache_data(show_spinner=False)
def replay_dates():
    return dd.replay_dates()


@st.cache_data(ttl=600, show_spinner=False)
def trend_live(event_ticker):
    return dd.market_trend_live(event_ticker)


@st.cache_data(show_spinner=False)
def trend_replay(event_ticker, home_abbr, tip):
    try:
        return dd.market_trend_replay(event_ticker, home_abbr, tip)
    except Exception:
        return pd.DataFrame(columns=['TIME_ET', 'PROB'])


@st.cache_data(ttl=3600, show_spinner=False)
def upcoming_live():
    try:
        return dd.upcoming_live()
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────── shared pieces

def fmt_tip(t):
    return '' if t is None or pd.isna(t) else pd.Timestamp(t).strftime('%I:%M %p').lstrip('0')


def sidebar():
    try:
        state = pd.read_csv('data/team_state.csv', parse_dates=['LAST_GAME_DATE'])
        m, mm = load_metrics()
        m = m.set_index('model')
        shipped = m[m['shipped']].index[0]
        st.sidebar.markdown("### Model")
        st.sidebar.caption(f"{shipped}")
        st.sidebar.caption(f"Data through {state['LAST_GAME_DATE'].max():%b %d, %Y}")
        st.sidebar.caption(f"Walk-forward held-out: log loss {m.loc[shipped, 'log_loss']:.3f}, "
                           f"AUC {m.loc[shipped, 'roc_auc']:.3f} (Elo only {m.loc['Elo only (logistic)', 'log_loss']:.3f} / "
                           f"{m.loc['Elo only (logistic)', 'roc_auc']:.3f})")
        if mm is not None:
            mm = mm.set_index('model')
            st.sidebar.caption(f"Vs Kalshi on {int(mm['games'].iloc[0])} games: model {mm.loc[shipped, 'log_loss']:.3f}, "
                               f"market {mm.loc['Kalshi pre-tip-off price', 'log_loss']:.3f} (lower is better)")
    except (FileNotFoundError, KeyError, IndexError):
        st.sidebar.caption("Run data_pipeline.py and train.py to populate metrics")


def metric_tiles():
    try:
        m, mm = load_metrics()
    except FileNotFoundError:
        return
    m = m.set_index('model')
    shipped = m[m['shipped']].index[0]
    elo = m.loc['Elo only (logistic)']
    tiles = [
        ('Held-out log loss', f"{m.loc[shipped, 'log_loss']:.3f}", f"Elo only {elo['log_loss']:.3f} · lower is better"),
        ('Held-out AUC', f"{m.loc[shipped, 'roc_auc']:.3f}", f"Elo only {elo['roc_auc']:.3f}"),
        ('Accuracy', f"{m.loc[shipped, 'accuracy']:.1%}", '1,762 walk-forward games'),
    ]
    if mm is not None:
        mm = mm.set_index('model')
        tiles.append(('Vs Kalshi (log loss)', f"{mm.loc[shipped, 'log_loss']:.3f} / "
                      f"{mm.loc['Kalshi pre-tip-off price', 'log_loss']:.3f}",
                      f"model / market on {int(mm['games'].iloc[0])} games · market is better"))
    ui.html_block('<div class="cv-tiles">' + ''.join(
        f'<div class="cv-tile"><div class="k">{ui.esc(k)}</div><div class="v">{ui.esc(v)}</div>'
        f'<div class="s">{ui.esc(s)}</div></div>' for k, v, s in tiles) + '</div>')


def gap_pill(model, market):
    if market is None or pd.isna(market):
        return ui.pill('NO MARKET', 'neutral')
    gap = (model - market) * 100
    if abs(gap) >= 5:
        return ui.pill(f'GAP {gap:+.0f}', 'warn')
    return ui.pill('IN LINE', 'neutral')


# ─────────────────────────────────────────────────────────────── dashboard

def matchup_card(g, games):
    hw, hl = dd.team_record(games, g['HOME_TEAM'], g['GAME_DATE'])
    aw, al = dd.team_record(games, g['AWAY_TEAM'], g['GAME_DATE'])
    final = g.get('HOME_PTS') is not None
    if final:
        a_cls = 'win' if not g['HOME_WIN'] else ''
        h_cls = 'win' if g['HOME_WIN'] else ''
        middle = (f'<div><span class="cv-score {a_cls}">{g["AWAY_PTS"]}</span>'
                  f'<span class="cv-vs">&nbsp;–&nbsp;</span><span class="cv-score {h_cls}">{g["HOME_PTS"]}</span>'
                  f'<div class="cv-small" style="text-align:center">FINAL</div></div>')
    else:
        middle = '<div class="cv-vs">AT</div>'
    tip = fmt_tip(g.get('TIP_TIME_ET'))
    meta = ' · '.join(x for x in [f"{tip} ET" if tip else '', dd.arena_name(g['HOME_ABBR']),
                                  f"{pd.Timestamp(g['GAME_DATE']):%a %b %d, %Y}"] if x)
    body = f"""
    <div class="cv-matchup">
      <div class="cv-team">{ui.badge(g['AWAY_ABBR'], 'lg')}
        <div class="cv-team-name">{ui.esc(g['AWAY_TEAM'])}</div><div class="cv-team-rec">Away · {aw}-{al}</div></div>
      {middle}
      <div class="cv-team">{ui.badge(g['HOME_ABBR'], 'lg')}
        <div class="cv-team-name">{ui.esc(g['HOME_TEAM'])}</div><div class="cv-team-rec">Home · {hw}-{hl}</div></div>
    </div>
    <div class="cv-meta">{ui.esc(meta)}</div>"""
    ui.html_block(ui.card('Game matchup', body))


def probability_card(g):
    p, m = g['MODEL_HOME_PROB'], g.get('MARKET_HOME_PROB')
    has_m = m is not None and not pd.isna(m)
    fav = g['HOME_ABBR'] if p >= 0.5 else g['AWAY_ABBR']
    split = (f'<div class="cv-split"><div style="width:{(1 - p) * 100:.1f}%;background:{ui.AWAY_SIDE}"></div>'
             f'<div style="width:{p * 100:.1f}%;background:{ui.HOME_SIDE}"></div></div>'
             f'<div style="display:flex;justify-content:space-between" class="cv-small">'
             f'<span>{ui.esc(g["AWAY_ABBR"])} {ui.pct(1 - p)}</span><span>{ui.esc(g["HOME_ABBR"])} {ui.pct(p)}</span></div>')
    gap = '' if not has_m else f'{(p - m) * 100:+.1f}'
    cells = f"""
    <div class="cv-prob-grid">
      <div class="cv-prob-cell"><div class="cv-prob-label">Model · {ui.esc(g['HOME_ABBR'])} win</div>
        <div class="cv-prob-val" style="color:#8fbcf3">{ui.pct(p, 1)}</div></div>
      <div class="cv-prob-cell"><div class="cv-prob-label">Kalshi · {ui.esc(g['HOME_ABBR'])} win</div>
        <div class="cv-prob-val" style="color:#f19a75">{ui.pct(m, 1) if has_m else 'n/a'}</div></div>
      <div class="cv-prob-cell"><div class="cv-prob-label">Gap</div>
        <div class="cv-prob-val">{gap if has_m else 'n/a'}<span class="sub">{'pts' if has_m else ''}</span></div>
        <div style="margin-top:4px">{gap_pill(p, m)}</div></div>
    </div>{split}"""
    sub = f"Model favors {fav}"
    if g.get('HOME_WIN') is not None:
        winner = g['HOME_ABBR'] if g['HOME_WIN'] else g['AWAY_ABBR']
        right = (p >= 0.5) == bool(g['HOME_WIN'])
        sub += f" · {winner} won · " + ("model right" if right else "model wrong")
    ui.html_block(ui.card('Win probability', cells, sub=sub))


def trend_card(g):
    with st.container(key=f"card-trend"):
        if g['MODE'] == 'replay':
            t = trend_replay(g.get('EVENT_TICKER'), g['HOME_ABBR'], g.get('TIP_TIME_ET'))
            sub = 'Hourly Kalshi midpoint before tip-off'
        else:
            t = trend_live(g.get('EVENT_TICKER')) if g.get('EVENT_TICKER') else pd.DataFrame()
            sub = 'Logged Kalshi snapshots today'
        ui.card_title(f"Market trend · {g['HOME_ABBR']} win probability", sub)
        if t is None or len(t) < 2:
            ui.html_block('<div class="cv-muted">Not enough market snapshots yet. Each refresh or visit '
                          'to this page adds one.</div>')
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t['TIME_ET'], y=t['PROB'] * 100, mode='lines', name='Kalshi',
                                 line=dict(color=ui.MARKET, width=2),
                                 hovertemplate='%{x|%b %d %I:%M %p}<br>Kalshi %{y:.1f}%<extra></extra>'))
        fig.add_trace(go.Scatter(x=[t['TIME_ET'].min(), t['TIME_ET'].max()], y=[g['MODEL_HOME_PROB'] * 100] * 2,
                                 mode='lines', name='Model', line=dict(color=ui.MODEL, width=2, dash='dash'),
                                 hovertemplate='Model %{y:.1f}%<extra></extra>'))
        lo = min(t['PROB'].min(), g['MODEL_HOME_PROB']) * 100
        hi = max(t['PROB'].max(), g['MODEL_HOME_PROB']) * 100
        pad = max(4, (hi - lo) * 0.25)
        fig.update_yaxes(range=[max(0, lo - pad), min(100, hi + pad)], ticksuffix='%')
        fig.update_xaxes(tickformat='%-I %p<br>%b %d')
        fig.update_layout(hovermode='x unified')
        ui.plotly_layout(fig, height=230)
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)


FORM_ROWS = [  # (label, column, format, higher is better)
    ('Elo', 'PRE_GAME_ELO', '{:.0f}', True),
    ('Net pts / game (L10)', 'ROLLING_PLUS_MINUS', '{:+.1f}', True),
    ('eFG% (L10)', 'ROLLING_eFG_PCT', '{:.1%}', True),
    ('Turnover % (L10)', 'ROLLING_TOV_PCT', '{:.1%}', False),
    ('Off. rebound % (L10)', 'ROLLING_ORB_PCT', '{:.1%}', True),
    ('Pts allowed /100 (L10)', 'ROLLING_DEF_RATING', '{:.1f}', False),
    ('Wins in last 5', 'WIN_STREAK', '{:.0f}', True),
]


def form_card(g):
    if g['MODE'] == 'replay':
        h, a = dd.team_form_replay(g['GAME_ID'], g['HOME_ID']), dd.team_form_replay(g['GAME_ID'], g['AWAY_ID'])
    else:
        state = load_predictor().state
        h, a = dd.team_form_live(state, g['HOME_TEAM']), dd.team_form_live(state, g['AWAY_TEAM'])
    if h is None or a is None:
        ui.html_block(ui.card('Team form', '<div class="cv-muted">No form data for this game.</div>'))
        return
    rows = [f'<div class="cv-form-head">{ui.badge(g["AWAY_ABBR"])}'
            f'<span class="cv-small">before tip-off</span>{ui.badge(g["HOME_ABBR"])}</div>']
    for label, col, fmt, higher in FORM_ROWS:
        hv, av = h.get(col), a.get(col)
        if hv is None or av is None or pd.isna(hv) or pd.isna(av):
            continue
        lo, hi = min(hv, av), max(hv, av)
        span = (hi - lo) or 1.0
        # bar length: share of the pair's combined range, so the better side is visibly longer
        base = lo - span
        aw, hw = (av - base) / (hi - base) * 100, (hv - base) / (hi - base) * 100
        a_better = (av > hv) if higher else (av < hv)
        h_better = (hv > av) if higher else (hv < av)
        rows.append(
            f'<div class="cv-form-label">{ui.esc(label)}</div>'
            f'<div class="cv-form-row">'
            f'<span class="cv-form-val {"better" if a_better else ""}">{fmt.format(av)}</span>'
            f'<div class="cv-track left"><div style="width:{aw:.0f}%;background:{ui.AWAY_SIDE if a_better else "rgba(147,160,184,0.45)"}"></div></div>'
            f'<div class="cv-track"><div style="width:{hw:.0f}%;background:{ui.HOME_SIDE if h_better else "rgba(147,160,184,0.45)"}"></div></div>'
            f'<span class="cv-form-val {"better" if h_better else ""}" style="text-align:right">{fmt.format(hv)}</span>'
            f'</div>')
    ui.html_block(ui.card('Team form', ''.join(rows), sub='colored bar = better side'))


def injury_card(g):
    def team_block(abbr, listed):
        if listed is None:
            return (f'<div style="margin-bottom:8px">{ui.badge(abbr)} '
                    f'<span class="cv-small">no report for this game</span></div>')
        g_league = [x for x in listed if str(x[1]).startswith('G League')]
        listed = [x for x in listed if not str(x[1]).startswith('G League')]
        gl_note = f'<div class="cv-small">+{len(g_league)} G League / two-way</div>' if g_league else ''
        if not listed:
            return (f'<div style="margin-bottom:8px">{ui.badge(abbr)} <span class="cv-small">nobody listed out</span>'
                    f'{gl_note}</div>')
        items = ''.join(
            f'<div class="cv-inj"><span class="cv-inj-name">{ui.esc(name)}</span>'
            f'<span class="cv-inj-reason">{ui.esc(str(reason).replace("Injury/Illness-", "") or "Out")}</span></div>'
            for name, reason in listed[:8])
        more = f'<div class="cv-small">+{len(listed) - 8} more</div>' if len(listed) > 8 else ''
        return f'<div style="margin-bottom:10px">{ui.badge(abbr)}<div style="margin-top:4px">{items}{more}{gl_note}</div></div>'

    sub = 'Out, last report before tip' if g['MODE'] == 'replay' else 'Out, latest report'
    ui.html_block(ui.card('Injury report', team_block(g['AWAY_ABBR'], g.get('AWAY_OUT')) +
                          team_block(g['HOME_ABBR'], g.get('HOME_OUT')), sub=sub))


def markets_card(games_today, selected_id, mode):
    rows = []
    for g in games_today:
        p, m = g['MODEL_HOME_PROB'], g.get('MARKET_HOME_PROB')
        has_m = m is not None and not pd.isna(m)
        cls = ' class="sel"' if g['GAME_ID'] == selected_id else (' class="hl"' if has_m and abs(p - m) >= 0.05 else '')
        result = ''
        if mode == 'replay':
            right = (p >= 0.5) == bool(g['HOME_WIN'])
            result = f'<td>{ui.pill("✓ RIGHT" if right else "✗ WRONG", "good" if right else "crit")}</td>'
        rows.append(
            f'<tr{cls}><td class="cv-game">{ui.badge(g["AWAY_ABBR"], "sm")} <span class="cv-small">@</span> {ui.badge(g["HOME_ABBR"], "sm")}</td>'
            f'<td class="num" style="color:#8fbcf3;font-weight:700">{ui.pct(p)}</td>'
            f'<td class="num" style="color:#f19a75;font-weight:700">{ui.pct(m) if has_m else "n/a"}</td>'
            f'<td>{gap_pill(p, m)}</td>{result}</tr>')
    head = ('<tr><th>Game</th><th class="num">Model</th><th class="num">Kalshi</th><th>Gap</th>'
            + ('<th>Model pick</th>' if mode == 'replay' else '') + '</tr>')
    body = f'<table class="cv-table">{head}{"".join(rows)}</table>' \
           f'<div class="cv-small" style="margin-top:8px">Home-team win probability. Kalshi = midpoint of the ' \
           f'YES bid/ask on its NBA game market (the prices behind PrizePicks game picks). Gaps under 5 points ' \
           f'are within fees and noise; on 1,223 past games the market was more accurate than the model.</div>'
    ui.html_block(ui.card('Game markets · Kalshi', body, sub=f'{len(games_today)} games'))


def drivers_card(g):
    with st.container(key="card-drivers"):
        ui.card_title('What drives the model', f"toward {g['HOME_ABBR']} (right) or {g['AWAY_ABBR']} (left)")
        if not g.get('FEATURES'):
            ui.html_block('<div class="cv-muted">No feature row for this game.</div>')
            return
        c = dd.contributions(load_model(), g['FEATURES']).head(7).iloc[::-1]
        pts = c['contribution'] * 25  # slope of the logistic curve at 50%: approx. win-probability points
        colors = [ui.HOME_SIDE if v >= 0 else ui.AWAY_SIDE for v in pts]
        fig = go.Figure(go.Bar(
            x=pts, y=c['label'], orientation='h', marker=dict(color=colors, cornerradius=4),
            customdata=np.stack([c['value']], axis=-1),
            hovertemplate='%{y}<br>≈ %{x:+.1f} pts of win probability<br>raw difference %{customdata[0]:.3g}<extra></extra>'))
        fig.add_vline(x=0, line_color='rgba(147,160,184,0.4)', line_width=1)
        fig.update_xaxes(ticksuffix=' pts')
        ui.plotly_layout(fig, height=250, legend=False)
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
        note = 'Approximate effect of each home-minus-away difference, from the current model.'
        if g['MODE'] == 'replay':
            note += ' The replay probability itself came from the walk-forward model trained before that date.'
        ui.html_block(f'<div class="cv-small">{note}</div>')


def upcoming_card(mode, current_date, all_dates):
    if mode.lower() == 'replay':
        later = [d for d in all_dates if d > current_date]
        if not later:
            ui.html_block(ui.card('Next game day', '<div class="cv-muted">Last day of the test period.</div>'))
            return
        nxt = later[0]
        games_next = replay_view(str(nxt.date()))
        rows = ''.join(
            f'<tr><td class="cv-game">{ui.badge(g["AWAY_ABBR"], "sm")} <span class="cv-small">@</span> {ui.badge(g["HOME_ABBR"], "sm")}</td>'
            f'<td class="num" style="color:#8fbcf3;font-weight:700">{ui.pct(g["MODEL_HOME_PROB"])}</td></tr>'
            for g in games_next[:6])
        ui.html_block(ui.card('Next game day', f'<table class="cv-table">{rows}</table>', sub=f'{nxt:%a %b %d}'))
        return
    up = upcoming_live()
    if not up:
        ui.html_block(ui.card('Coming up', '<div class="cv-muted">No regular-season games in the next 3 days. '
                                           'The 2026-27 season starts October 20.</div>'))
        return
    abbr = dd.team_abbrs()
    rows = ''.join(
        f'<tr><td class="cv-small">{pd.Timestamp(r["GAME_DATE"]):%a %b %d}</td>'
        f'<td class="cv-game">{ui.badge(abbr.get(r["AWAY_TEAM"], "?"), "sm")} <span class="cv-small">@</span> {ui.badge(abbr.get(r["HOME_TEAM"], "?"), "sm")}</td>'
        f'<td class="cv-small">{fmt_tip(r.get("TIP_TIME_ET"))}</td></tr>' for r in up[:8])
    ui.html_block(ui.card('Coming up', f'<table class="cv-table">{rows}</table>'))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_REFRESH_STEPS = [
    ('Downloading new games from stats.nba.com', 'src/ingest.py'),
    ('Rebuilding features', 'src/data_pipeline.py'),
    ('Retraining and re-running the walk-forward evaluation', 'src/train.py'),
    ('Archiving the latest injury report', 'src/injury_reports.py'),
]


def run_script(path):
    """Run one pipeline script in a separate process (fresh imports); raise with its output on failure."""
    import subprocess
    proc = subprocess.run([sys.executable, '-u', path], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    if proc.returncode != 0:
        tail = '\n'.join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"{path} failed:\n{tail}")
    return proc.stdout


def refresh_bar():
    """Data freshness and the two refresh buttons."""
    try:
        state = pd.read_csv('data/team_state.csv', parse_dates=['LAST_GAME_DATE'])
        data_through = f"{state['LAST_GAME_DATE'].max():%b %d, %Y}"
    except FileNotFoundError:
        data_through = 'no data'
    trained = (datetime.fromtimestamp(os.path.getmtime('models/nba_model.joblib')).strftime('%b %d, %I:%M %p')
               if os.path.exists('models/nba_model.joblib') else 'not trained')
    report = load_predictor().report
    report_txt = 'none' if report is None else f"{pd.Timestamp(report['REPORT_TIME'].max()):%b %d, %I:%M %p} ET"

    info, b1, b2 = st.columns([3.2, 1, 1], vertical_alignment='center')
    with info:
        ui.html_block(f'<div class="cv-small">Games through <b>{data_through}</b> · model trained {trained} · '
                      f'latest injury report {report_txt}</div>')
    with b1:
        quick = st.button('↻ Update injuries & odds', use_container_width=True,
                          help='Re-download the latest injury report and Kalshi prices, recompute today\'s '
                               'predictions and log them. Takes a few seconds. Use it shortly before tip-off.')
    with b2:
        full = st.button('⟳ Full data refresh', use_container_width=True,
                         help='Download new game results, rebuild features and retrain the model. Takes a few '
                              'minutes. Use it the morning after games.')
    if quick:
        load_predictor.clear()
        live_view.clear()
        trend_live.clear()
        upcoming_live.clear()
        st.session_state['refresh_msg'] = ('success', 'Injury report and Kalshi prices updated; today\'s '
                                                      'predictions were re-logged.')
        st.rerun()
    if full:
        with st.status('Refreshing data...', expanded=True) as status:
            try:
                for label, path in FULL_REFRESH_STEPS:
                    status.write(f"{label}...")
                    if path == 'src/injury_reports.py':
                        try:
                            run_script(path)       # offseason or report outage is not fatal
                        except RuntimeError as e:
                            status.write(f"Skipped: {str(e).splitlines()[-1]}")
                        continue
                    run_script(path)
            except RuntimeError as e:
                status.update(label='Refresh failed', state='error')
                st.error(str(e))
                return
            status.update(label='Refresh complete', state='complete')
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state['refresh_msg'] = ('success', 'Data refreshed and model retrained.')
        st.rerun()
    msg = st.session_state.pop('refresh_msg', None)
    if msg:
        st.toast(msg[1], icon='✅')


def page_dashboard():
    games_all = load_games()
    try:
        live = live_view()
    except Exception as e:
        live = []
        st.warning(f"Live slate unavailable: {e}")
    dates = replay_dates()
    date_list = list(dates.index)

    # Header: brand + date, mode switch, replay date
    head_l, head_r = st.columns([3, 2], vertical_alignment='bottom')
    default_mode = 'Live' if live else 'Replay'
    with head_r:
        c1, c2 = st.columns([1, 1.6], vertical_alignment='bottom')
        with c1:
            mode = st.segmented_control('Mode', ['Live', 'Replay'], default=default_mode, key='mode',
                                        label_visibility='collapsed') or default_mode
        with c2:
            if mode == 'Replay':
                labels = {d: f"{d:%a %b %d, %Y} · {int(dates.loc[d, 'games'])} games" for d in reversed(date_list)}
                day = st.selectbox('Replay day', list(labels), format_func=labels.get, key='replay_day',
                                   label_visibility='collapsed')
    if mode == 'Live':
        games_today = live
        day = pd.Timestamp.now().normalize()
        subtitle = 'Live · predictions are logged before tip-off'
    else:
        games_today = replay_view(str(pd.Timestamp(day).date()))
        subtitle = 'Replay · walk-forward predictions made before these games, Kalshi price at tip-off'
    with head_l:
        ui.html_block(f'<div class="cv-header"><div><div class="cv-brand">🏀 NBA Outcome Predictor</div>'
                      f'<div class="cv-date">{pd.Timestamp(day):%B %d, %Y}</div>'
                      f'<div class="cv-mode">{ui.esc(subtitle)}</div></div></div>')
    refresh_bar()

    if not games_today:
        metric_tiles()
        ui.html_block(ui.card('No games', '<div class="cv-muted">No regular-season games today. '
                                          'Switch to <b>Replay</b> to explore a past game day.</div>'))
        upcoming_card('Live', None, date_list)
        return

    ids = [g['GAME_ID'] for g in games_today]
    if st.session_state.get('game') not in ids:
        st.session_state['game'] = ids[0]

    left, mid, right = st.columns([1.15, 2.25, 1.6], gap='medium')
    with left:
        with st.container(key='card-games'):
            ui.card_title("Today's games" if mode == 'Live' else 'Games', 'model · home win %')
            for g in games_today:
                if g.get('HOME_PTS') is not None:
                    detail = f"{g['AWAY_PTS']}–{g['HOME_PTS']}"
                else:
                    detail = fmt_tip(g.get('TIP_TIME_ET')) or g.get('STATUS', '')
                label = f"{g['AWAY_ABBR']} @ {g['HOME_ABBR']} · {detail} · {ui.pct(g['MODEL_HOME_PROB'])}"
                if st.button(label, key=f"game-{g['GAME_ID']}",
                             type='primary' if g['GAME_ID'] == st.session_state['game'] else 'secondary'):
                    st.session_state['game'] = g['GAME_ID']
                    st.rerun()
    g = next(x for x in games_today if x['GAME_ID'] == st.session_state['game'])

    with mid:
        matchup_card(g, games_all)
        probability_card(g)
        trend_card(g)
        c1, c2 = st.columns([1, 1], gap='small')
        with c1:
            form_card(g)
        with c2:
            injury_card(g)
    with right:
        markets_card(games_today, g['GAME_ID'], 'replay' if mode == 'Replay' else 'live')
        drivers_card(g)
        upcoming_card('Replay' if mode == 'Replay' else 'Live', pd.Timestamp(day), date_list)


# ─────────────────────────────────────────────────────────────── markets

def page_markets():
    ui.html_block('<div class="cv-header"><div><div class="cv-brand">Markets</div>'
                  '<div class="cv-date">Model vs Kalshi</div>'
                  '<div class="cv-mode">Held-out games from the walk-forward test, with Kalshi\'s price at tip-off</div></div></div>')
    metric_tiles()
    preds = load_test_predictions()
    market = pd.read_csv('data/market_history.csv', dtype={'GAME_ID': str})
    market['GAME_ID'] = market['GAME_ID'].astype(int)
    d = preds.merge(market[['GAME_ID', 'MARKET_HOME_PROB']], on='GAME_ID').dropna(subset=['MARKET_HOME_PROB'])
    if d.empty:
        st.info("Run `python src/market_odds.py --history` and `python src/train.py` first.")
        return
    c1, c2 = st.columns([1.2, 1], gap='medium')
    with c1:
        with st.container(key='card-scatter'):
            ui.card_title('Model vs market, game by game', f'{len(d)} games · home win probability')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Agree',
                                     line=dict(color='rgba(147,160,184,0.45)', width=1, dash='dot'), hoverinfo='skip'))
            fig.add_trace(go.Scatter(
                x=d['MARKET_HOME_PROB'] * 100, y=d['MODEL_PROB'] * 100, mode='markers', name='Game',
                marker=dict(color=ui.MODEL, size=8, opacity=0.55, line=dict(color='#131c2e', width=1)),
                customdata=np.stack([d['GAME_DATE'].dt.strftime('%b %d, %Y'), d['TARGET']], axis=-1),
                hovertemplate='%{customdata[0]}<br>Kalshi %{x:.0f}% · model %{y:.0f}%<br>home won: %{customdata[1]}<extra></extra>'))
            fig.update_xaxes(title_text='Kalshi home win %', range=[0, 100], ticksuffix='%')
            fig.update_yaxes(title_text='Model home win %', range=[0, 100], ticksuffix='%')
            ui.plotly_layout(fig, height=380, legend=False)
            st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
    with c2:
        gap = (d['MODEL_PROB'] - d['MARKET_HOME_PROB']) * 100
        big = d[gap.abs() >= 5]
        def ll(p, y):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        model_right = ((big['MODEL_PROB'] >= 0.5) == big['TARGET']).mean() if len(big) else np.nan
        market_right = ((big['MARKET_HOME_PROB'] >= 0.5) == big['TARGET']).mean() if len(big) else np.nan
        model_ll, market_ll = ll(big['MODEL_PROB'], big['TARGET']), ll(big['MARKET_HOME_PROB'], big['TARGET'])
        verdict = ("On these games the market's probabilities were more accurate (lower log loss), even though "
                   "both picked a similar share of winners." if market_ll < model_ll else
                   "On these games the model's probabilities were at least as accurate as the market's.")
        body = (f'<table class="cv-table">'
                f'<tr><th>Gap of 5+ points</th><th class="num">Model</th><th class="num">Kalshi</th></tr>'
                f'<tr><td>Games</td><td class="num" colspan="2">{len(big)} of {len(d)}</td></tr>'
                f'<tr><td>Picked the winner</td><td class="num">{model_right:.1%}</td>'
                f'<td class="num">{market_right:.1%}</td></tr>'
                f'<tr><td>Log loss (lower is better)</td><td class="num">{model_ll:.3f}</td>'
                f'<td class="num">{market_ll:.3f}</td></tr></table>'
                f'<div class="cv-small" style="margin-top:8px">{verdict} Median gap over all {len(d)} games: '
                f'{gap.abs().median():.1f} pts.</div>')
        ui.html_block(ui.card('Big disagreements', body))
        with st.container(key='card-gaphist'):
            ui.card_title('Size of the gap', 'model minus Kalshi, points')
            fig = go.Figure(go.Histogram(x=gap, nbinsx=40, marker=dict(color=ui.MODEL, line=dict(color='#131c2e', width=1)),
                                         hovertemplate='%{x} pts: %{y} games<extra></extra>'))
            fig.add_vline(x=0, line_color='rgba(147,160,184,0.5)', line_width=1)
            ui.plotly_layout(fig, height=200, legend=False)
            st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)


# ─────────────────────────────────────────────────────────────── custom predictor

def page_predictor():
    ui.html_block('<div class="cv-header"><div><div class="cv-brand">Predictor</div>'
                  '<div class="cv-date">Any matchup, today</div>'
                  '<div class="cv-mode">Players listed Out on today\'s injury report are pre-checked; change them to test scenarios</div></div></div>')
    refresh_bar()
    predictor = load_predictor()
    teams = predictor.teams()
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox('Home team', teams, index=teams.index('Boston Celtics') if 'Boston Celtics' in teams else 0)
    with c2:
        away = st.selectbox('Away team', teams, index=teams.index('Los Angeles Lakers') if 'Los Angeles Lakers' in teams else 1)

    report_time = None if predictor.report is None else pd.Timestamp(predictor.report['REPORT_TIME'].max())
    if report_time is not None and report_time.date() == datetime.now().date():
        st.caption(f"Official injury report: {report_time:%b %d, %I:%M %p} ET.")
    elif report_time is not None:
        st.caption(f"No injury report for today's games yet (latest archived: {report_time:%b %d, %Y}). "
                   "Check injured players manually.")

    outs = {}
    cols = st.columns(2, gap='medium')
    for side, team, col in [('home', home, cols[0]), ('away', away, cols[1])]:
        with col:
            with st.container(key=f'card-rot-{side}'):
                ui.card_title(f"{'Home' if side == 'home' else 'Away'} · {team}", 'tick = out')
                listed = set(predictor.report_out(team) or [])
                outs[side] = []
                for i, p in enumerate(predictor.rotation(team).to_dict('records')):
                    star = '⭐ ' if i < 4 else ''
                    if st.checkbox(f"{star}{p['PLAYER_NAME']} ({p['impact_score']:.1f})"
                                   + (' · report: OUT' if p['PLAYER_NAME'] in listed else ''),
                                   value=p['PLAYER_NAME'] in listed, key=f"{side}_out_{team}_{i}"):
                        outs[side].append(p['PLAYER_NAME'])

    if st.button('Predict', type='primary', use_container_width=True):
        r = predictor.predict(home, away, outs['home'], outs['away'])
        if r['stale_warning']:
            st.warning(r['stale_warning'])
        abbr = dd.team_abbrs()
        g = {'MODE': 'live', 'MODEL_HOME_PROB': r['home_prob'], 'MARKET_HOME_PROB': None,
             'HOME_ABBR': abbr.get(home, home[:3]), 'AWAY_ABBR': abbr.get(away, away[:3]),
             'FEATURES': r['features'].iloc[0].to_dict()}
        probability_card(g)
        drivers_card(g)


# ─────────────────────────────────────────────────────────────── track record

def page_track_record():
    import prediction_log
    ui.html_block('<div class="cv-header"><div><div class="cv-brand">Track record</div>'
                  '<div class="cv-date">Live predictions</div>'
                  '<div class="cv-mode">The last prediction logged before each tip-off, scored once results are in</div></div></div>')
    record = prediction_log.track_record()
    if record.empty:
        ui.html_block(ui.card('Nothing to score yet', (
            '<div class="cv-muted">No finished games with a logged prediction. Predictions are logged by '
            '<code>python src/daily_slate.py</code> (the refresh scripts run it) and by the Dashboard on game '
            'days; results arrive with <code>python src/ingest.py</code>. The 2026-27 regular season starts '
            'October 20.</div>')))
        return
    tiles = []
    for name, col in [('Model', 'MODEL_HOME_PROB'), ('Kalshi', 'MARKET_HOME_PROB')]:
        r = record.dropna(subset=[col])
        if len(r):
            m = prediction_log.running_metrics(r, col).iloc[-1]
            tiles.append((f'{name} log loss', f"{m['log_loss']:.3f}", f"{len(r)} games · accuracy {m['accuracy']:.1%}"))
    ui.html_block('<div class="cv-tiles">' + ''.join(
        f'<div class="cv-tile"><div class="k">{ui.esc(k)}</div><div class="v">{v}</div><div class="s">{ui.esc(s)}</div></div>'
        for k, v, s in tiles) + '</div>')
    with st.container(key='card-running'):
        ui.card_title('Running log loss', 'lower is better')
        fig = go.Figure()
        for name, col, color in [('Model', 'MODEL_HOME_PROB', ui.MODEL), ('Kalshi', 'MARKET_HOME_PROB', ui.MARKET)]:
            m = prediction_log.running_metrics(record, col)
            if len(m):
                fig.add_trace(go.Scatter(x=m['games'], y=m['log_loss'], mode='lines', name=name,
                                         line=dict(color=color, width=2),
                                         hovertemplate=f'{name}: %{{y:.3f}} after %{{x}} games<extra></extra>'))
        fig.add_hline(y=np.log(2), line_dash='dot', line_color='rgba(147,160,184,0.5)',
                      annotation_text='coin flip', annotation_font_color=ui.INK_3)
        fig.update_xaxes(title_text='Games')
        ui.plotly_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
    if len(record) < 100:
        st.caption(f"Only {len(record)} games so far: these numbers will move a lot until a few hundred are in.")


# ─────────────────────────────────────────────────────────────── model

def page_model():
    from sklearn.metrics import roc_curve, roc_auc_score
    ui.html_block('<div class="cv-header"><div><div class="cv-brand">Model</div>'
                  '<div class="cv-date">Walk-forward performance</div>'
                  '<div class="cv-mode">Every number is from games the model had not seen when it predicted them</div></div></div>')
    metric_tiles()
    m, mm = load_metrics()
    preds = load_test_predictions()

    def table(df, cols, names):
        head = ''.join(f'<th class="{"num" if i else ""}">{ui.esc(n)}</th>' for i, n in enumerate(names))
        body = ''
        for _, r in df.iterrows():
            cells = ''.join(f'<td class="{"num" if i else ""}">{r[c] if not i else ui.esc(r[c])}</td>'
                            for i, c in enumerate(cols))
            body += f'<tr>{cells}</tr>'
        return f'<table class="cv-table"><tr>{head}</tr>{body}</table>'

    fmt = m.copy()
    for c in ['roc_auc', 'log_loss', 'brier']:
        fmt[c] = fmt[c].map(lambda v: f'{v:.4f}')
    fmt['accuracy'] = fmt['accuracy'].map(lambda v: f'{v:.1%}')
    fmt['model'] = [f"{n} ★" if s else n for n, s in zip(m['model'], m['shipped'])]
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        ui.html_block(ui.card('Walk-forward results', table(fmt, ['model', 'roc_auc', 'accuracy', 'log_loss', 'brier'],
                                                             ['Model', 'AUC', 'Accuracy', 'Log loss', 'Brier']),
                              sub=f'{len(preds)} held-out games · ★ shipped'))
    with c2:
        if mm is not None:
            f2 = mm.copy()
            for c in ['roc_auc', 'log_loss', 'brier']:
                f2[c] = f2[c].map(lambda v: f'{v:.4f}')
            f2['accuracy'] = f2['accuracy'].map(lambda v: f'{v:.1%}')
            ui.html_block(ui.card('Against the market', table(f2, ['model', 'roc_auc', 'accuracy', 'log_loss', 'brier'],
                                                               ['Source', 'AUC', 'Accuracy', 'Log loss', 'Brier']),
                                  sub=f"{int(mm['games'].iloc[0])} games, {mm['first_game'].iloc[0]} to {mm['last_game'].iloc[0]}"))

    c1, c2, c3 = st.columns(3, gap='medium')
    y, p = preds['TARGET'], preds['MODEL_PROB']
    with c1:
        with st.container(key='card-calib'):
            ui.card_title('Calibration', 'predicted vs actual home win rate')
            bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
            cal = preds.assign(b=pd.cut(p, bins=bins, include_lowest=True)).groupby('b', observed=True).agg(
                n=('TARGET', 'size'), pred=('MODEL_PROB', 'mean'), act=('TARGET', 'mean'))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Perfect',
                                     line=dict(color='rgba(147,160,184,0.45)', dash='dot', width=1), hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=cal['pred'] * 100, y=cal['act'] * 100, mode='lines+markers', name='Model',
                                     line=dict(color=ui.MODEL, width=2), marker=dict(size=9),
                                     customdata=cal['n'],
                                     hovertemplate='predicted %{x:.0f}% · actual %{y:.0f}%<br>%{customdata} games<extra></extra>'))
            fig.update_xaxes(range=[0, 100], ticksuffix='%')
            fig.update_yaxes(range=[0, 100], ticksuffix='%')
            ui.plotly_layout(fig, height=280, legend=False)
            st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
    with c2:
        with st.container(key='card-roc'):
            fpr, tpr, _ = roc_curve(y, p)
            ui.card_title('ROC curve', f'AUC {roc_auc_score(y, p):.3f}')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='rgba(147,160,184,0.45)', dash='dot', width=1),
                                     hoverinfo='skip', name='Chance'))
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=ui.MODEL, width=2), name='Model',
                                     hovertemplate='false positive %{x:.2f}<br>true positive %{y:.2f}<extra></extra>'))
            ui.plotly_layout(fig, height=280, legend=False)
            st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
    with c3:
        with st.container(key='card-weights'):
            ui.card_title('Feature weights', 'standardized logistic coefficients')
            w = feature_weights(load_model()).rename(index=dd.FEATURE_LABELS)
            w = w.reindex(w.abs().sort_values().index)
            fig = go.Figure(go.Bar(x=w.values, y=w.index, orientation='h',
                                   marker=dict(color=[ui.HOME_SIDE if v >= 0 else ui.AWAY_SIDE for v in w.values], cornerradius=4),
                                   hovertemplate='%{y}: %{x:+.3f}<extra></extra>'))
            fig.add_vline(x=0, line_color='rgba(147,160,184,0.4)', line_width=1)
            ui.plotly_layout(fig, height=280, legend=False)
            st.plotly_chart(fig, use_container_width=True, config=ui.PLOTLY_CONFIG)
    st.caption("Positive weights push toward the home team. Weights are for standardized home-minus-away "
               "differences, so a negative weight on 'Pts allowed /100' means allowing more points hurts.")


# ─────────────────────────────────────────────────────────────── navigation

sidebar()
pages = [
    st.Page(page_dashboard, title='Dashboard', icon=':material/sports_basketball:', url_path='dashboard', default=True),
    st.Page(page_markets, title='Markets', icon=':material/show_chart:', url_path='markets'),
    st.Page(page_predictor, title='Predictor', icon=':material/tune:', url_path='predictor'),
    st.Page(page_track_record, title='Track Record', icon=':material/fact_check:', url_path='track-record'),
    st.Page(page_model, title='Model', icon=':material/insights:', url_path='model'),
]
st.navigation(pages, position='top').run()
