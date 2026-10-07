"""
Frozen copy of the shipped models for the 2026-27 paper test (docs/paper_trading_plan.md).

On FREEZE_DATE (or the first refresh after it) the trained win-probability and spread models are
copied to models/frozen/ with a manifest, and never refit. Live predictions keep using
models/nba_model.joblib, which retrains daily and may change at the monthly reviews; the slate
logs both, so they can be compared on the same games. The frozen model's inputs (Elo, form,
injuries) still update with every game: only its weights stay fixed. A 2025-26 replay found fixed
weights about as accurate as daily refits, so the frozen copy is a fair yardstick.

    python src/frozen_model.py                          # status
    python src/frozen_model.py --freeze-on 2026-10-19   # freeze if today is on/after the date and
                                                        # nothing is frozen yet (the refresh runs this)
    python src/frozen_model.py --freeze                 # freeze now (refuses if already frozen)
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone
import pandas as pd

FREEZE_DATE = '2026-10-19'
FROZEN_DIR = 'models/frozen'
MODEL_PATH = 'models/nba_model.joblib'
SPREAD_PATH = 'models/spread_model.joblib'
ET = 'America/New_York'


def _manifest_path(frozen_dir):
    return os.path.join(frozen_dir, 'manifest.json')


def status(frozen_dir=FROZEN_DIR):
    """The manifest dict, or None if nothing is frozen."""
    path = _manifest_path(frozen_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def freeze(frozen_dir=FROZEN_DIR, model_path=MODEL_PATH, spread_path=SPREAD_PATH, today=None):
    """Copy the current models into `frozen_dir` with a manifest. Never overwrites a frozen model."""
    from matchups import FEATURES
    from prediction_log import model_version
    if status(frozen_dir) is not None:
        raise FileExistsError(f'{frozen_dir} already holds a frozen model; delete it by hand to re-freeze')
    os.makedirs(frozen_dir, exist_ok=True)
    shutil.copy2(model_path, os.path.join(frozen_dir, 'nba_model.joblib'))
    has_spread = os.path.exists(spread_path)
    if has_spread:
        shutil.copy2(spread_path, os.path.join(frozen_dir, 'spread_model.joblib'))
    manifest = {'frozen_on': str(pd.Timestamp(today or pd.Timestamp.now(tz=ET).date()).date()),
                'frozen_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': model_version(model_path),
                'spread_model': has_spread, 'features': list(FEATURES)}
    with open(_manifest_path(frozen_dir), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def freeze_on(date=FREEZE_DATE, frozen_dir=FROZEN_DIR, today=None, **paths):
    """Freeze if `today` (default: today, Eastern) is on or after `date` and nothing is frozen."""
    today = pd.Timestamp(today or pd.Timestamp.now(tz=ET).date())
    if status(frozen_dir) is not None or today < pd.Timestamp(date):
        return None
    return freeze(frozen_dir, today=today, **paths)


class FrozenModel:
    """The frozen win-probability (and spread) model, predicting from a live feature row."""

    def __init__(self, frozen_dir=FROZEN_DIR):
        import joblib
        self.manifest = status(frozen_dir)
        if self.manifest is None:
            raise FileNotFoundError(f'no frozen model in {frozen_dir}')
        self.version = self.manifest['model_version']
        self.features = self.manifest['features']
        self.model = joblib.load(os.path.join(frozen_dir, 'nba_model.joblib'))
        self.spread = None
        if self.manifest.get('spread_model'):
            import spread_model
            self.spread = spread_model.load('spread', os.path.join(frozen_dir, 'spread_model.joblib'))

    def predict(self, features):
        """(home win probability, home margin or None, spread sigma or None) for a one-row DataFrame."""
        x = features[self.features]
        prob = float(self.model.predict_proba(x)[0][1])
        if self.spread is None:
            return prob, None, None
        return prob, float(self.spread.predict(features)[0]), self.spread.sigma


def load(frozen_dir=FROZEN_DIR):
    """FrozenModel, or None before the freeze."""
    try:
        return FrozenModel(frozen_dir)
    except FileNotFoundError:
        return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--freeze-on', default=None, metavar='DATE')
    args = ap.parse_args()
    if args.freeze:
        print('Frozen:', freeze())
    elif args.freeze_on:
        m = freeze_on(args.freeze_on)
        print('Frozen:', m) if m else print(f'Not frozen now (already frozen, or before {args.freeze_on}).')
    s = status()
    print('Frozen model:', 'none yet' if s is None else
          f"version {s['model_version']}, frozen on {s['frozen_on']} ({len(s['features'])} features)")
