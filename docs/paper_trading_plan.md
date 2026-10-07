# Paper-trading plan, 2026-27 regular season

Hypothetical analysis, not betting advice. Paper trading only: this project places no orders and
contains no trading code. Kalshi prices stay local (git-ignored); this file contains rules only,
no market figures.

## Why

On last season's held-out games, no strategy showed a reliable edge after fees. Two ideas were
positive but were found by looking at those same games, so they are hypotheses. This plan writes
the rules down **before** the season starts (2026-10-20), so the live results are a clean test.

## Rules (frozen until the end of the regular season)

Every rule uses the **last** prediction the slate logged at least 30 minutes before that game's
tip-off, and the Kalshi price logged with it (the price you could have acted on), one bet per
game, regular-season games only (no preseason, play-in or playoffs). This matches the backtest,
which used the last injury report published 30+ minutes before tip-off. The slate therefore runs
repeatedly on game evenings (see Implementation), not once in the morning.

| ID | Strategy | Rule | Role |
|---|---|---|---|
| S1 | Model picks the market underdog | The model's favorite is priced under 50c | **Primary hypothesis** |
| S2 | Big spread lean | Spread decision is a lean with edge >= 10 points after fees | Exploratory |
| S3 | Market favorite, every game | The team priced over 50c | Baseline (no model) |
| S3a | Heavy market favorite | Market favorite priced 90c or more | Exploratory |
| S4 | Model pick, every game | The model's favorite | Reference |
| S5 | 3-pick lineup, model beats market | PrizePicks-style lineup of 3 legs; each leg is the model's pick (model probability >= 50%) and the model's probability is above the market midpoint for that team | Exploratory |
| S6 | 6-pick lineup, model beats market | Same legs as S5, 6 per lineup | Exploratory (watch the fee drag) |

S5 and S6 legs are taken in tip-off order: one lineup per day from the first 3 (S5) or first 6
(S6) qualifying games; no lineup on days with fewer. Stake $1 per lineup.

Frozen: the 50c / 90c cutoffs, the 10-point spread threshold, the S5/S6 leg rule and lineup
sizes, stake and pricing rules below. Changing any of these restarts the count for the affected
strategy, with a dated note here. The model itself may improve during the season, under the
rules in the next section.

## Season horizon and model changes

The plan runs for one season: from 2026-10-20 to the last regular-season game (April 2027),
then a final review ("cash out" on paper). The model is expected to improve along the way, so
two models are logged side by side:

- **Frozen model:** a copy of the trained win-probability and spread models saved on
  2026-10-19 and never refit. Its inputs (Elo, form, injuries) still update with every game;
  only its weights stay fixed (the replay below found fixed weights about as accurate as daily
  refits). Its picks are the clean test of the rules above, and the report judges the decision
  rules on it.
- **Live model:** the model as it improves. Every pick is tagged with its `MODEL_VERSION`.

The live model refits after every game day on all games so far, with every season weighted
equally. A replay of 2025-26 (trained on earlier seasons, refit before each game day) found
daily refits about equal to the frozen pre-season fit (log loss -0.0006, SE 0.0010), and giving
the current season extra weight (x2 to x8) was worse on the 2024-25 replay used to choose it.
The model's accuracy rises late in the season for both versions alike, because late-season games
are easier to predict, not because the model learned more.

Rules for changing the live model:
1. A change ships only if it passes the existing walk-forward rule (lower held-out log loss than
   the model it replaces, recorded in RESULTS.md), measured on games before the change date.
2. Changes are made only at monthly review points (about the 15th of each month), not day to
   day, so results are not chased.
3. Live results from this season are never used to pick the strategy cutoffs above.
4. At each review, compare the live and frozen models on the same games played since the last
   change: log loss and each strategy's P&L. A change that helps on past data but trails the
   frozen model live is noted, not reverted mid-month.

What one season can and cannot show: with no edge at all, a strategy still ends a season ahead
about half the time, and even a real edge of a few cents per dollar often ends behind. So the
end-of-season result alone will not settle whether there is an edge; closing line value and the
frozen-vs-live comparison are the more informative outputs.

## How each bet is priced (three venues)

| Venue | Price per $1 of payout |
|---|---|
| Kalshi single | ask + Kalshi taker fee (0.07 x p x (1 - p), rounded up per order) |
| PrizePicks single | ask + Kalshi fee + 2c PrizePicks fee |
| PrizePicks lineup | (product of the legs' asks) + Kalshi fee + 2c, charged once on the combined price |

The PrizePicks formulas are reverse-engineered from published and in-app multipliers (October
2026) and are estimates; see "Checks" below.

**Lineups.** Built mechanically so there is no after-the-fact choosing: each day, the legs of a
strategy sorted by tip-off time (then game ID); one lineup of the first k legs for k = 2, 3, 4
and 6 when the day has that many. Different games only.

## What is measured

Per strategy and venue:
- bets, win rate vs average price paid
- P&L per $1 and its standard error; ROI on money staked
- running P&L, maximum drawdown and worst losing streak on a $100 paper bankroll (flat $1 per bet)
- **closing line value (CLV):** the tip-off midpoint minus the logged entry midpoint for the side
  taken, in cents. If the market keeps moving toward our side before tip-off, the model is seeing
  something real. CLV is much less noisy than wins and losses, so it can say something within
  weeks; P&L needs seasons.

## Decision rules (written in advance)

Judged only on games from 2026-10-20 on.

| Strategy | Stop ("dead") | Promising (not proven) |
|---|---|---|
| S1 | After 100 bets, P&L <= -2c per $1 | After 150+ bets, P&L > 0 with t >= 2, or after 50+ bets, CLV > 0 with t >= 2 |
| S2, S3a | After 100 bets, P&L <= -2c per $1 | Same as S1; still needs another season to confirm, since these are exploratory |
| S3, S4 | Not judged; they are the yardsticks | |
| S5 | After 100 lineups, return <= -12% per $1 (about what fees alone cost) | After 100+ lineups, return > 0 with t >= 2 |
| S6 | Not judged this season: too few days have 6 qualifying games | Watch only |

Even "promising" is not proof: about 400-500 bets are needed to confirm an edge of 6-7c, i.e.
several seasons for S1 (about 1 game in 9 qualifies). Lineups are noisier still: a lineup
usually loses its whole stake or pays several times it, so even 100 lineups leave a wide margin.

## Checks (manual, local)

PrizePicks has no public price feed, so occasionally record from the app into
`data/prizepicks_checks.csv` (git-ignored): date and time, game, team, single multiplier, and for
lineups the legs and the payout shown for a stake. These confirm or correct the pricing formulas.
Building a lineup to read its payout does not require submitting it.

## Implementation

- Already logged by `daily_slate.py` / `prediction_log.py`: model and market probabilities, bid
  and ask, the spread decision and edge, and the model's pick with `PICK_MARKET_UNDERDOG`. S1-S6
  can be computed from the existing columns (S5/S6 from `PICK_P`, `PICK_MID` and `PICK_FILL`, with
  tip-off order from `TIP_TIME_ET`); no new log columns are needed.
- Done: `scripts/slate.bat` / `slate.sh` (injury report + slate only, for repeated runs).
- Done: `src/paper_report.py` applies these rules to the log and writes the git-ignored
  `data/paper_report.md`; the morning refresh rebuilds it.
- Done: `python src/market_odds.py --tip-prices` saves each logged game's Kalshi price at
  scheduled tip-off (last 1-minute candle, else the last hourly one) to the git-ignored
  `data/tip_prices.csv`; the morning refresh runs it before the report, and the report's CLV
  uses it (falling back to the last pre-tip-off logged price).
- Done: `src/frozen_model.py`. The morning refresh copies the models to the git-ignored
  `models/frozen/` (with a manifest: date, version, features) on the first run on or after
  2026-10-19, and never overwrites them. The slate logs the frozen model's probability, margin,
  pick and spread decision next to the live model's (`FROZEN_*` columns, a new log segment;
  earlier rows untouched). The report shows both models and a monthly live / frozen / market
  comparison for the reviews. Back up `models/frozen/` if you reinstall or move machines.
- Timing (checked 2026-10-06): no scheduled task runs the refresh or the slate on this machine,
  and the GitHub workflow is manual-only and does not run the slate. The refresh scripts suggest
  10:00 AM, but at that hour the slate uses an old injury report: in 2025-26, between the first
  evening report (about 5:30 PM ET) and the last report 30+ minutes before tip-off, a
  Questionable or Doubtful player was moved to Out in about 28% of team-games. Plan: the full
  refresh (ingest, train) once in the morning, and the slate alone every 30 minutes from 11:00 AM
  to 10:30 PM ET on game days (weekend games can tip at noon), using `scripts/slate.bat` (or
  `slate.sh`), which writes to the git-ignored `logs/slate.log`. The log is append-only and the
  track record already uses the last pre-tip-off row per game, so repeated runs are safe.
- Days or games the slate misses are missing, not backfilled with later prices.
