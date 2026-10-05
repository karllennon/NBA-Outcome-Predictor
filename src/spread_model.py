"""
Point-spread model (and the structure for other numeric targets like game totals).

A target is described by a TargetConfig: which training column it predicts, which features it
uses, how to build the regression, and where the fitted model is saved. Adding another target
(e.g. total points) means adding a column to the training set and an entry to TARGETS; the
fitting, uncertainty, saving, loading and probability code below is shared.

Spread convention (betting): the home team's line is minus its predicted margin, so
home -4.5 means the home team is predicted to win by 4.5. Lines are displayed rounded to the
nearest 0.5, favorite first ("BOS -4.5"), "PK" when the predicted margin rounds to zero.

Uncertainty: sigma is the standard deviation of residuals on the most recent SIGMA_HOLDOUT of the
training games, predicted by a model fit on the earlier training games; the model is then refit
on all training games. It never sees test games. P(margin > x) = 1 - Phi((x - mu) / sigma).
"""
from dataclasses import dataclass, field
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matchups import FEATURES

SIGMA_HOLDOUT = 0.2


def make_ridge(alpha=1.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


@dataclass
class TargetConfig:
    name: str
    target: str                       # training-set column to predict
    features: list = field(default_factory=lambda: list(FEATURES))
    make_model: object = make_ridge
    model_path: str = ''


TARGETS = {
    # Home point margin (home score minus away score); the spread is its negative.
    'spread': TargetConfig('spread', 'MARGIN', model_path='models/spread_model.joblib'),
    # Later, e.g.: 'total': TargetConfig('total', 'TOTAL_POINTS', model_path='models/total_model.joblib'),
}


@dataclass
class FittedTarget:
    config: TargetConfig
    model: object
    sigma: float

    def predict(self, X):
        return self.model.predict(X[self.config.features])

    def prob_over(self, X, threshold):
        """P(target > threshold) for each row."""
        mu = self.predict(X)
        return 1 - norm.cdf((np.asarray(threshold, dtype=float) - mu) / self.sigma)


def fit_target(config, train, sigma_holdout=SIGMA_HOLDOUT):
    """Fit on all training rows; sigma from a holdout at the end of the training rows only."""
    train = train.dropna(subset=[config.target])
    cut = int(len(train) * (1 - sigma_holdout))
    early, late = train.iloc[:cut], train.iloc[cut:]
    resid = late[config.target] - config.make_model().fit(early[config.features], early[config.target]).predict(
        late[config.features])
    sigma = float(np.sqrt(np.mean(resid ** 2)))
    model = config.make_model().fit(train[config.features], train[config.target])
    return FittedTarget(config, model, sigma)


def save(fitted, path=None):
    joblib.dump({'name': fitted.config.name, 'model': fitted.model, 'sigma': fitted.sigma,
                 'features': fitted.config.features, 'target': fitted.config.target},
                path or fitted.config.model_path)


def load(name='spread', path=None):
    config = TARGETS[name]
    d = joblib.load(path or config.model_path)
    cfg = TargetConfig(config.name, d['target'], list(d['features']), config.make_model, config.model_path)
    return FittedTarget(cfg, d['model'], d['sigma'])


# ---------------------------------------------------------------- spread conventions

def round_half(x):
    """Nearest 0.5 (display)."""
    return np.round(np.asarray(x, dtype=float) * 2) / 2


def home_spread(home_margin):
    """Betting-convention home line: negative when the home team is favored."""
    return -np.asarray(home_margin, dtype=float)


def format_spread(home_margin, home_abbr, away_abbr):
    """'BOS -4.5' (favorite and its line, rounded to 0.5) or 'PK'."""
    line = float(round_half(abs(home_margin)))
    if line == 0:
        return 'PK'
    fav = home_abbr if home_margin > 0 else away_abbr
    return f"{fav} -{line:g}"


def prob_home_covers(home_margin, sigma, home_line):
    """
    P(home covers home_line), home_line in betting convention (home -4.5 -> -4.5): the home team
    must win by more than -home_line, i.e. P(margin > -home_line).
    """
    return 1 - norm.cdf((-np.asarray(home_line, dtype=float) - np.asarray(home_margin, dtype=float)) / sigma)


def prob_margin_over(home_margin, sigma, threshold):
    """P(home margin > threshold)."""
    return 1 - norm.cdf((np.asarray(threshold, dtype=float) - np.asarray(home_margin, dtype=float)) / sigma)
