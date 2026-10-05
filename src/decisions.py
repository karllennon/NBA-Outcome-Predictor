"""
Hypothetical decisions from model probabilities and Kalshi prices (paper trading only).

This module only does arithmetic on prices; it places no orders and contains no trading code.

For a binary market (YES pays $1):
    fee(P)      Kalshi taker fee per contract at price P: 0.07 x P x (1 - P), rounded up to the
                cent (Kalshi fee schedule, "most markets", checked October 2026)
    edge        model probability - market midpoint - fee, in percentage points
    EV          expected profit per $1 contract = model probability - price - fee
A side is a "lean" when its edge after fees is at least MIN_EDGE_PTS; otherwise "No play".
Paper trades are recorded at the ask (the price you could actually buy at), which is
more conservative than the midpoint used for the edge.

Hypothetical analysis, not betting advice.
"""
import math
import numpy as np

MIN_EDGE_PTS = 5.0          # minimum edge after fees to call a lean, percentage points
LARGE_EDGE_PTS = 15.0       # edges above this get a "check late news" warning
KALSHI_TAKER_RATE = 0.07
DISCLAIMER = 'Hypothetical analysis, not betting advice.'


def fee(price, contracts=1):
    """Kalshi taker fee per contract at `price` (dollars), rounded up to the cent per order."""
    if price is None or not (0 < price < 1):
        return 0.0
    total = math.ceil(round(KALSHI_TAKER_RATE * contracts * price * (1 - price) * 100, 6)) / 100
    return total / contracts


def side_edge(p_model, price):
    """(edge in points after fees, EV per $1 contract) for buying a side at `price`."""
    f = fee(price)
    ev = p_model - price - f
    return ev * 100, ev


def decide(p_yes, yes_bid, yes_ask, min_edge=MIN_EDGE_PTS):
    """
    Best side of one binary market. YES costs the YES midpoint; NO costs 1 - YES midpoint.
    Returns dict: side ('YES', 'NO' or None), edge_pts, ev, p_side (model chance for the side),
    mid (side's midpoint price), fill (side's ask: YES ask, or 1 - YES bid for NO), fee.
    """
    if p_yes is None or yes_bid is None or yes_ask is None or any(
            isinstance(x, float) and math.isnan(x) for x in (p_yes, yes_bid, yes_ask)):
        return {'side': None, 'best_side': None, 'edge_pts': None, 'ev': None, 'p_side': None, 'mid': None,
                'fill': None, 'fee': None, 'reason': 'no market price'}
    mid = (yes_bid + yes_ask) / 2
    options = []
    for side, p, price, fill in [('YES', p_yes, mid, yes_ask), ('NO', 1 - p_yes, 1 - mid, 1 - yes_bid)]:
        edge, ev = side_edge(p, price)
        options.append({'side': side, 'edge_pts': edge, 'ev': ev, 'p_side': p, 'mid': price, 'fill': fill,
                        'fee': fee(price)})
    best = max(options, key=lambda o: o['edge_pts'])
    best['best_side'] = best['side']
    if best['edge_pts'] < min_edge:
        return {**best, 'side': None, 'reason': f"best edge {best['edge_pts']:+.1f} pts < {min_edge:g}"}
    return {**best, 'reason': 'edge after fees'}


def bad_price_flags(p_home, home_mid, home_abbr, away_abbr):
    """
    'Likely winner, bad price': a team the model favors whose market price is above the model's
    probability (e.g. 80 cents for a 75% chance). Returns a list of messages.
    """
    if home_mid is None or (isinstance(home_mid, float) and math.isnan(home_mid)):
        return []
    out = []
    for team, p, price in [(home_abbr, p_home, home_mid), (away_abbr, 1 - p_home, 1 - home_mid)]:
        if p > 0.5 and price > p:
            out.append(f"Likely winner, bad price: {team} {p:.0%} to win but costs {price * 100:.0f}¢")
    return out


def settle(side, fill, won):
    """Profit/loss of a $1 paper contract bought at `fill` (plus fee); None if no trade."""
    if side is None or fill is None:
        return None
    return (1.0 if won else 0.0) - fill - fee(fill)


def spread_market_prob(home_margin, sigma, fav_is_home, strike):
    """
    Model chance that the favorite's 'wins by over STRIKE' market pays: P(fav margin > strike).
    """
    from spread_model import prob_margin_over
    if fav_is_home:
        return float(prob_margin_over(home_margin, sigma, strike))
    return float(1 - prob_margin_over(home_margin, sigma, -strike))


def game_decisions(g, min_edge=MIN_EDGE_PTS):
    """
    Decisions for one game dict (live or replay view): moneyline and spread.
    Uses MODEL_HOME_PROB, MARKET home YES bid/ask, MODEL_HOME_MARGIN, SPREAD_SIGMA and the
    market spread's main line (MARKET_SPREAD_INFO). Returns dict with 'moneyline', 'spread',
    'flags' and 'warnings'.
    """
    home, away = g['HOME_ABBR'], g['AWAY_ABBR']
    out = {'moneyline': None, 'spread': None, 'flags': [], 'warnings': []}

    bid, ask = g.get('MARKET_YES_BID', g.get('HOME_YES_BID')), g.get('MARKET_YES_ASK', g.get('HOME_YES_ASK'))
    ml = decide(g['MODEL_HOME_PROB'], bid, ask, min_edge)
    if ml['mid'] is not None:
        team = home if ml['best_side'] == 'YES' else away      # the side with the better edge
        ml.update({'team': team, 'label': f"Lean {team}" if ml['side'] else 'No play',
                   'line_text': f"{home} {((bid + ask) / 2) * 100:.0f}¢"})
        out['flags'] += bad_price_flags(g['MODEL_HOME_PROB'], (bid + ask) / 2, home, away)
    out['moneyline'] = ml

    info = g.get('MARKET_SPREAD_INFO')
    margin, sigma = g.get('MODEL_HOME_MARGIN'), g.get('SPREAD_SIGMA')
    if info and margin is not None and sigma and info.get('MAIN_STRIKE') == info.get('MAIN_STRIKE') \
            and info.get('MAIN_STRIKE') is not None:
        fav = info['FAV']
        strike = float(info['MAIN_STRIKE'])
        p_yes = spread_market_prob(margin, sigma, fav == home, strike)
        sp = decide(p_yes, info.get('MAIN_YES_BID'), info.get('MAIN_YES_ASK'), min_edge)
        dog = away if fav == home else home
        team, pick = (fav, f"{fav} -{strike:g}") if sp['best_side'] == 'YES' else (dog, f"{dog} +{strike:g}")
        sp.update({'team': team, 'pick': pick, 'label': f"Lean {pick}" if sp['side'] else 'No play'})
        sp.update({'fav': fav, 'strike': strike, 'p_fav_covers': p_yes,
                   'line_text': f"{fav} -{strike:g} at {((info['MAIN_YES_BID'] + info['MAIN_YES_ASK']) / 2) * 100:.0f}¢"
                   if sp['mid'] is not None else f"{fav} -{strike:g}"})
        out['spread'] = sp
    for key in ('moneyline', 'spread'):
        d = out[key]
        if d and d.get('side') and d['edge_pts'] > LARGE_EDGE_PTS:
            out['warnings'].append(f"{key.capitalize()} edge over {LARGE_EDGE_PTS:g} pts: unusually large, "
                                   "check late injury news before trusting it")
    return out


def reason(g, d, market):
    """One plain-English line explaining a decision."""
    home, away = g['HOME_ABBR'], g['AWAY_ABBR']
    margin = g.get('MODEL_HOME_MARGIN')
    by = '' if margin is None else (f"Model: {home if margin > 0 else away} by {abs(margin):.0f}. ")
    if market == 'spread':
        if not d or d.get('mid') is None:
            return 'No usable Kalshi spread price.'
        side_text = d['pick']
        price = d['mid']
        return (f"{by}Line: {side_text} at {price * 100:.0f}¢. Model gives {d['p_side']:.0%} to cover, "
                f"so you'd pay {price * 100:.0f}¢ for a {d['p_side']:.0%} chance"
                f" ({d['edge_pts']:+.1f} pts after fees).")
    if not d or d.get('mid') is None:
        return 'No usable Kalshi moneyline price.'
    team = d['team']
    return (f"Model: {team} {d['p_side']:.0%} to win. Price: {d['mid'] * 100:.0f}¢, "
            f"so you'd pay {d['mid'] * 100:.0f}¢ for a {d['p_side']:.0%} chance ({d['edge_pts']:+.1f} pts after fees).")
