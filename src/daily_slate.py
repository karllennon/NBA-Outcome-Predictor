"""
Today's games: model probability, Kalshi market probability and the gap, with every
pre-tip-off prediction appended to data/prediction_log.csv. Shared by the CLI and the
dashboard's Model vs Market page so both show and log the same numbers.

    python src/daily_slate.py [--date YYYY-MM-DD] [--no-log]
"""
import argparse
from datetime import datetime
import pandas as pd

GAP_HIGHLIGHT = 0.05   # model vs market gaps this large (5 points) are highlighted
REGULAR_SEASON, PLAYOFFS = '002', '004'


def market_spreads(game_date, snapshot=True):
    """{(home abbr, away abbr): market spread info} from Kalshi's spread ladders (local snapshot)."""
    import market_odds
    try:
        if snapshot:
            ladders = market_odds.spread_snapshot(game_date)
        else:
            day = pd.Timestamp(game_date)
            frames = []
            for ev in market_odds.open_spread_events():
                d, away, home = market_odds.parse_event_ticker(ev['event_ticker'])
                if d == day:
                    frames.append(market_odds._ladder_rows(ev.get('markets', [])).assign(HOME=home, AWAY=away))
            ladders = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as e:
        print(f"[!] Kalshi spread markets unavailable: {type(e).__name__}: {e}")
        return {}
    out = {}
    if ladders is None or ladders.empty:
        return out
    for (home, away), lad in ladders.groupby(['HOME', 'AWAY']):
        info = market_odds.market_spread_from_ladder(lad, home, away)
        if info:
            out[(home, away)] = info
    return out


def spread_fields(res, market_spread, home_abbr, away_abbr):
    """Model spread (home line, betting convention), market line, and the toss-up flag."""
    import spread_model
    margin = res.get('home_margin')
    market_text = (None if not market_spread else
                   spread_model.format_spread(-market_spread['HOME_LINE'], home_abbr, away_abbr))
    base = {'HOME_ABBR': home_abbr, 'AWAY_ABBR': away_abbr, 'MARKET_SPREAD_TEXT': market_text}
    if margin is None:
        return {**base, 'MODEL_HOME_MARGIN': None, 'SPREAD': None, 'SPREAD_TEXT': None, 'SPREAD_SIGMA': None,
                'MARKET_SPREAD': None, 'MARKET_SPREAD_INFO': market_spread, 'TOSS_UP': False}
    return {**base, 'MODEL_HOME_MARGIN': margin, 'SPREAD': float(spread_model.home_spread(margin)),
            'SPREAD_TEXT': spread_model.format_spread(margin, home_abbr, away_abbr),
            'SPREAD_SIGMA': res.get('spread_sigma'),
            'MARKET_SPREAD': None if not market_spread else market_spread['HOME_LINE'],
            'MARKET_SPREAD_INFO': market_spread,
            # win probability and spread pick different favorites: essentially a toss-up
            'TOSS_UP': (res['home_prob'] > 0.5) != (margin > 0)}


def _utc_to_et(ts):
    return pd.Timestamp(ts).tz_localize('UTC').tz_convert('America/New_York').tz_localize(None)


def decision_fields(row, report_time=None):
    """Hypothetical decisions (decisions.py) for one slate row, flattened for the log."""
    import decisions as D
    d = D.game_decisions(row)
    warnings = list(d['warnings'])
    if report_time is not None and row.get('MARKET_TIME_ET') is not None and report_time > row['MARKET_TIME_ET']:
        warnings.append('Injury report updated after the market price was taken: check late news first')
    out = {'DECISIONS': d, 'FLAGS': d['flags'], 'WARNINGS': warnings, 'MIN_EDGE_PTS': D.MIN_EDGE_PTS}
    for key, prefix in (('moneyline', 'ML'), ('spread', 'SPREAD')):
        x = d[key] or {}
        out.update({f'{prefix}_DECISION': x.get('label'), f'{prefix}_SIDE': x.get('side'),
                    f'{prefix}_P': x.get('p_side'), f'{prefix}_MID': x.get('mid'), f'{prefix}_FILL': x.get('fill'),
                    f'{prefix}_EDGE_PTS': x.get('edge_pts'), f'{prefix}_EV': x.get('ev')})
    out['ML_TEAM'] = (d['moneyline'] or {}).get('team')
    out['SPREAD_FAV'] = (d['spread'] or {}).get('fav')
    out['SPREAD_STRIKE'] = (d['spread'] or {}).get('strike')
    return out


def build_slate(game_date=None, predictor=None, log=True, snapshot=True):
    """
    One row per regular-season or playoff game on `game_date` (default today, Eastern):
    GAME_ID, TIP_TIME_ET, STATUS, HOME_TEAM, AWAY_TEAM, MODEL_HOME_PROB, MARKET_HOME_PROB,
    GAP (model minus market), plus the players counted out. Games already under way or
    finished are shown but never logged.
    """
    from inference import GamePredictor
    from schedule import games_on
    import market_odds
    import prediction_log

    ET = 'America/New_York'
    game_date = pd.Timestamp(game_date or pd.Timestamp.now(tz=ET).date())
    sched = games_on(game_date)
    sched = sched[sched['GAME_ID'].str[:3].isin([REGULAR_SEASON, PLAYOFFS])]
    if sched.empty:
        return pd.DataFrame()

    if predictor is None:
        predictor = GamePredictor()
        predictor.load_injury_report()

    try:
        market = market_odds.snapshot(game_date) if snapshot else market_odds.games_on(game_date)
        market = market.set_index(['HOME_TEAM', 'AWAY_TEAM'])
    except Exception as e:  # Kalshi down: show the model alone
        print(f"[!] Kalshi unavailable: {type(e).__name__}: {e}")
        market = pd.DataFrame()
    spreads = market_spreads(game_date, snapshot)
    abbr = {name: code for code, name in market_odds.team_names().items()}

    now_et = pd.Timestamp.now(tz=ET).tz_localize(None)
    version = prediction_log.model_version()
    report_time = (None if predictor.report is None else
                   pd.Timestamp(predictor.report['REPORT_TIME'].max()))
    rows, to_log = [], []
    for g in sched.itertuples():
        try:
            res = predictor.predict(g.HOME_TEAM, g.AWAY_TEAM, game_date=game_date)
        except ValueError as e:
            print(f"[!] {g.AWAY_TEAM} @ {g.HOME_TEAM}: {e}")
            continue
        m = market.loc[(g.HOME_TEAM, g.AWAY_TEAM)] if (g.HOME_TEAM, g.AWAY_TEAM) in market.index else None
        market_prob = None if m is None else m['MARKET_HOME_PROB']
        row = {'GAME_ID': g.GAME_ID, 'GAME_DATE': game_date.date(), 'TIP_TIME_ET': g.TIP_TIME_ET,
               'STATUS': g.STATUS, 'HOME_TEAM': g.HOME_TEAM, 'AWAY_TEAM': g.AWAY_TEAM,
               'MODEL_HOME_PROB': res['home_prob'], 'MARKET_HOME_PROB': market_prob,
               'MARKET_YES_BID': None if m is None else m['HOME_YES_BID'],
               'MARKET_YES_ASK': None if m is None else m['HOME_YES_ASK'],
               'HOME_OUT': '; '.join(res['home_out']), 'AWAY_OUT': '; '.join(res['away_out']),
               'EVENT_TICKER': None if m is None else m['EVENT_TICKER'],
               'FEATURES': res['features'].iloc[0].to_dict(),
               'HOME_DETAILS': res['home_details'], 'AWAY_DETAILS': res['away_details']}
        row['GAP'] = None if pd.isna(market_prob) else res['home_prob'] - market_prob
        row.update(spread_fields(res, spreads.get((abbr.get(g.HOME_TEAM), abbr.get(g.AWAY_TEAM))),
                                 abbr.get(g.HOME_TEAM, g.HOME_TEAM[:3]), abbr.get(g.AWAY_TEAM, g.AWAY_TEAM[:3])))
        row['MARKET_TIME_ET'] = None if m is None else _utc_to_et(m['SNAPSHOT_TIME_UTC'])
        row.update(decision_fields(row, report_time))
        rows.append(row)
        pregame = g.STATUS == 'scheduled' and (pd.isna(g.TIP_TIME_ET) or now_et < g.TIP_TIME_ET)
        if log and pregame:
            to_log.append({**row, 'MODEL_VERSION': version,
                           'INPUTS': prediction_log.inputs_json(res['features'])})
    if to_log:
        prediction_log.append(to_log)
    return pd.DataFrame(rows)


def slate_table(slate):
    """Display table for the dashboard, with gaps of 5+ points highlighted (pandas Styler)."""
    view = pd.DataFrame({
        'Tip (ET)': [None if pd.isna(t) else f"{pd.Timestamp(t):%I:%M %p}" for t in slate['TIP_TIME_ET']],
        'Game': slate['AWAY_TEAM'] + ' @ ' + slate['HOME_TEAM'],
        'Status': slate['STATUS'],
        'Model (home)': slate['MODEL_HOME_PROB'],
        'Kalshi (home)': pd.to_numeric(slate['MARKET_HOME_PROB'], errors='coerce'),
        'Gap (pts)': pd.to_numeric(slate['GAP'], errors='coerce') * 100,
        'Out (home / away)': slate['HOME_OUT'] + ' / ' + slate['AWAY_OUT'],
    })

    def highlight(row):
        big = pd.notna(row['Gap (pts)']) and abs(row['Gap (pts)']) >= GAP_HIGHLIGHT * 100
        return ['background-color: rgba(255, 196, 0, 0.25)' if big else '' for _ in row]

    return view.style.apply(highlight, axis=1).format(
        {'Model (home)': '{:.1%}', 'Kalshi (home)': '{:.1%}', 'Gap (pts)': '{:+.1f}'}, na_rep='n/a')


def print_slate(slate):
    if slate.empty:
        print("No regular-season or playoff games on this date.")
        return
    for r in slate.itertuples():
        market = '   n/a' if pd.isna(r.MARKET_HOME_PROB) else f"{r.MARKET_HOME_PROB:6.1%}"
        gap = '' if pd.isna(r.GAP) else f"{r.GAP * 100:+5.1f} pts" + ('  <-- gap >= 5' if abs(r.GAP) >= GAP_HIGHLIGHT else '')
        tip = '' if pd.isna(r.TIP_TIME_ET) else f"{pd.Timestamp(r.TIP_TIME_ET):%I:%M %p}"
        print(f"{tip:>8}  {r.AWAY_TEAM:>24} @ {r.HOME_TEAM:<24} model {r.MODEL_HOME_PROB:6.1%}  "
              f"market {market}  {gap}")
        if r.SPREAD_TEXT:
            print(f"{'':>10}spread: model {r.SPREAD_TEXT:<10} market {r.MARKET_SPREAD_TEXT or 'n/a'}"
                  + ("   (win probability and spread disagree: close to a toss-up)" if r.TOSS_UP else ''))
        for label, dec, edge in [('moneyline', r.ML_DECISION, r.ML_EDGE_PTS),
                                 ('spread', r.SPREAD_DECISION, r.SPREAD_EDGE_PTS)]:
            if dec:
                print(f"{'':>10}{label}: {dec}" + ('' if edge is None or pd.isna(edge) else f" ({edge:+.1f} pts after fees)"))
        for note in list(r.FLAGS or []) + list(r.WARNINGS or []):
            print(f"{'':>10}! {note}")
    print("\nProbabilities are for the home team. Gaps under ~5 points are within Kalshi's fees "
          "and normal model error; even larger gaps are often the market knowing something the "
          "model doesn't (late scratches, rest).")
    import decisions
    print(f"Leans need an edge of at least {decisions.MIN_EDGE_PTS:g} points after Kalshi fees. "
          f"{decisions.DISCLAIMER} Paper trades only; no orders are placed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None)
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()
    print(f"--- NBA slate for {args.date or datetime.now().strftime('%Y-%m-%d')} ---")
    print_slate(build_slate(args.date, log=not args.no_log))
