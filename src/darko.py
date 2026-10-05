"""
DARKO DPM snapshots (darko.app leaderboard CSVs saved as data/darko/darko_<YYYY-MM-DD>.csv).

Each snapshot is the leaderboard "as of" its date. A game only ever uses the latest snapshot
dated strictly before it. Only the as-of columns are read (DPM and projected minutes); the
"DPM now" column, which is today's rating, would leak the future and is ignored.

Caveat (RESULTS.md, Phase 3.3): DARKO's Time Machine shows ratings recomputed by today's model
over games up to each date, not necessarily the ratings published on that date.
"""
import os
import re
import numpy as np
import pandas as pd

DARKO_DIR = 'data/darko'
REPLACEMENT_DPM = -2.0   # roughly a fringe NBA player


def _key(name):
    from injury_reports import normalize_name
    return normalize_name(name)


class DarkoRatings:
    def __init__(self, directory=DARKO_DIR):
        frames = []
        for f in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
            m = re.match(r'darko_(\d{4}-\d{2}-\d{2})\.csv$', f)
            if not m:
                continue
            d = pd.read_csv(os.path.join(directory, f), encoding='utf-8-sig')
            frames.append(pd.DataFrame({
                'AS_OF': pd.Timestamp(m.group(1)), 'KEY': d['Player'].map(_key),
                'DPM': pd.to_numeric(d['DPM'], errors='coerce'),
                'MPG': pd.to_numeric(d['MPG'], errors='coerce')}))
        self.snapshots = sorted({f['AS_OF'].iloc[0] for f in frames})
        all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['AS_OF', 'KEY', 'DPM', 'MPG'])
        self.by_date = {d: g.drop_duplicates('KEY').set_index('KEY') for d, g in all_rows.groupby('AS_OF')}

    def snapshot_before(self, game_date):
        """Latest snapshot date strictly before the game, or None."""
        game_date = pd.Timestamp(game_date).normalize()
        earlier = [d for d in self.snapshots if d < game_date]
        return earlier[-1] if earlier else None

    def value(self, player_name, game_date):
        """
        (DPM - replacement) x share of the 48 minutes, from the latest snapshot before the game.
        Players missing from that snapshot (mostly rookies before their first games) count as
        replacement level (0). None when no snapshot precedes the game.
        """
        snap = self.snapshot_before(game_date)
        if snap is None:
            return None
        table = self.by_date[snap]
        key = _key(player_name)
        if key not in table.index:
            return 0.0
        row = table.loc[key]
        dpm = row['DPM'] if pd.notna(row['DPM']) else REPLACEMENT_DPM
        mpg = row['MPG'] if pd.notna(row['MPG']) else 0.0
        return float(max(dpm - REPLACEMENT_DPM, 0.0) * mpg / 48.0)
