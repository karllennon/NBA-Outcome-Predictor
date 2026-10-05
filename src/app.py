import streamlit as st
import pandas as pd
import joblib
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from inference import GamePredictor
from matchups import FEATURES
from train import feature_weights
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

st.set_page_config(
    page_title="NBA Outcome Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(ttl=900)
def load_predictor():
    predictor = GamePredictor()
    predictor.load_injury_report()  # re-downloaded at most every 15 minutes
    return predictor

@st.cache_data
def load_test_predictions():
    return pd.read_csv('data/test_predictions.csv', parse_dates=['GAME_DATE'])

@st.cache_resource
def load_model():
    return joblib.load('models/nba_model.joblib')

def get_all_teams():
    return load_predictor().teams()

def get_rotation(team_name):
    return load_predictor().rotation(team_name).to_dict('records')

def get_report_out(team_name):
    return set(load_predictor().report_out(team_name) or [])

@st.cache_data(ttl=600)
def get_todays_games():
    """(home, away) for today's regular-season and playoff games, or None."""
    try:
        from schedule import games_on
        games = games_on()
        games = games[games['GAME_ID'].str[:3].isin(['002', '004'])]
        matchups = [(h, a) for h, a in zip(games['HOME_TEAM'], games['AWAY_TEAM']) if h and a]
        return matchups or None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner="Building today's slate (model, injury report, Kalshi)...")
def get_slate():
    """Model vs market for today's games; logs pre-tip-off predictions (at most every 10 min)."""
    from daily_slate import build_slate
    return build_slate(predictor=load_predictor())

def run_prediction(home_team, away_team, home_injuries, away_injuries):
    result = load_predictor().predict(home_team, away_team, home_injuries, away_injuries)

    def describe(details):
        return [f"🔴 **{player}**: {status.upper()} (-{impact:.1f})" for player, status, impact in details]

    return (result['home_prob'], describe(result['home_details']), describe(result['away_details']),
            result['injury_diff'], result['stale_warning'])

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/National_Basketball_Association_logo.svg/200px-National_Basketball_Association_logo.svg.png", width=80)
st.sidebar.title("NBA Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏀 Today's Slate", "💹 Model vs Market", "📈 Track Record",
                                     "📊 Backtest Results", "🤖 Model Performance"])
st.sidebar.markdown("---")
try:
    _state = pd.read_csv('data/team_state.csv', parse_dates=['LAST_GAME_DATE'])
    st.sidebar.caption(f"Data through: {_state['LAST_GAME_DATE'].max():%b %d, %Y}")
    _metrics = pd.read_csv('data/test_metrics.csv').set_index('model')
    _shipped = _metrics[_metrics['shipped']].index[0]
    _elo = _metrics.loc['Elo only (logistic)']
    st.sidebar.caption(f"Model: {_shipped}")
    st.sidebar.caption(f"Walk-forward held-out: log loss {_metrics.loc[_shipped, 'log_loss']:.3f}, "
                       f"AUC {_metrics.loc[_shipped, 'roc_auc']:.3f} "
                       f"(Elo only: {_elo['log_loss']:.3f} / {_elo['roc_auc']:.3f})")
    if os.path.exists('data/market_metrics.csv'):
        _mm = pd.read_csv('data/market_metrics.csv').set_index('model')
        st.sidebar.caption(f"Vs Kalshi on {int(_mm['games'].iloc[0])} games: model log loss "
                           f"{_mm.loc[_shipped, 'log_loss']:.3f}, market "
                           f"{_mm.loc['Kalshi pre-tip-off price', 'log_loss']:.3f}")
except (FileNotFoundError, KeyError):
    st.sidebar.caption("Run data_pipeline.py and train.py to populate metrics")

# ─────────────────────────────────────────────
# PAGE 1: TODAY'S SLATE
# ─────────────────────────────────────────────
if page == "🏀 Today's Slate":
    st.title("🏀 NBA Game Predictor")
    st.markdown(f"**{datetime.now().strftime('%A, %B %d, %Y')}**")
    st.markdown("---")

    all_teams = get_all_teams()

    todays_games = get_todays_games()

    if todays_games:
        st.success(f"Found {len(todays_games)} games today")
        game_options = [f"{away} @ {home}" for home, away in todays_games]
        selected_game = st.selectbox("Select a game", game_options)
        idx = game_options.index(selected_game)
        home_team = todays_games[idx][0]
        away_team = todays_games[idx][1]
    else:
        st.info("No regular-season games today (or the schedule is unavailable). Select teams manually.")
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("🏠 Home Team", all_teams, index=all_teams.index("Boston Celtics"))
        with col2:
            away_team = st.selectbox("✈️ Away Team", all_teams, index=all_teams.index("Los Angeles Lakers"))

    st.markdown("---")

    home_rotation = get_rotation(home_team)
    away_rotation = get_rotation(away_team)
    home_report_out, away_report_out = get_report_out(home_team), get_report_out(away_team)

    predictor = load_predictor()
    report_time = None if predictor.report is None else pd.Timestamp(predictor.report['REPORT_TIME'].max())
    if report_time is not None and report_time.date() == datetime.now().date():
        st.caption(f"Official injury report: {report_time:%b %d, %I:%M %p} ET. "
                   "Players listed Out are pre-checked; uncheck or check boxes to override.")
    elif report_time is not None:
        st.caption(f"No injury report for today's games yet (latest archived: {report_time:%b %d, %Y}). "
                   "Check injured players manually.")
    else:
        msg = f" ({predictor.report_error})" if predictor.report_error else ""
        st.caption(f"No official injury report available{msg}. Check injured players manually.")

    col1, col2 = st.columns(2)
    home_injuries = []
    away_injuries = []

    with col1:
        st.subheader(f"🏠 {home_team}")
        st.caption("✓ = OUT")
        if home_rotation:
            for i, player in enumerate(home_rotation):
                core_tag = "⭐ " if i < 4 else ""
                listed = player['PLAYER_NAME'] in home_report_out
                is_out = st.checkbox(
                    f"{core_tag}{player['PLAYER_NAME']} ({player['impact_score']:.1f})"
                    + (" 📋 report: OUT" if listed else ""),
                    value=listed,
                    key=f"home_out_{home_team}_{i}"
                )
                if is_out:
                    home_injuries.append(player['PLAYER_NAME'])
        else:
            st.warning("No rotation data found")

    with col2:
        st.subheader(f"✈️ {away_team}")
        st.caption("✓ = OUT")
        if away_rotation:
            for i, player in enumerate(away_rotation):
                core_tag = "⭐ " if i < 4 else ""
                listed = player['PLAYER_NAME'] in away_report_out
                is_out = st.checkbox(
                    f"{core_tag}{player['PLAYER_NAME']} ({player['impact_score']:.1f})"
                    + (" 📋 report: OUT" if listed else ""),
                    value=listed,
                    key=f"away_out_{away_team}_{i}"
                )
                if is_out:
                    away_injuries.append(player['PLAYER_NAME'])
        else:
            st.warning("No rotation data found")

    st.markdown("---")

    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing matchup..."):
            try:
                prob, home_details, away_details, injury_diff, stale_warning = run_prediction(
                    home_team, away_team, home_injuries, away_injuries
                )
                away_prob = 1 - prob
                if stale_warning:
                    st.warning(stale_warning)

                st.markdown("---")
                st.subheader("📊 Prediction Results")

                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric(f"🏠 {home_team}", f"{prob:.1%}",
                              delta="Favored" if prob > 0.5 else "Underdog")
                with res_col2:
                    st.metric("vs", "")
                with res_col3:
                    st.metric(f"✈️ {away_team}", f"{away_prob:.1%}",
                              delta="Favored" if away_prob > 0.5 else "Underdog")

                prob_df = pd.DataFrame({
                    'Team': [home_team, away_team],
                    'Probability': [prob, away_prob]
                })
                st.bar_chart(prob_df.set_index('Team'))

                if home_details or away_details:
                    st.markdown("**Injury Impact**")
                    if home_details:
                        st.markdown(f"*{home_team}:*")
                        for d in home_details:
                            st.markdown(f"&nbsp;&nbsp;{d}")
                    if away_details:
                        st.markdown(f"*{away_team}:*")
                        for d in away_details:
                            st.markdown(f"&nbsp;&nbsp;{d}")

                winner = home_team if prob > 0.5 else away_team
                confidence = max(prob, away_prob)
                if confidence > 0.65:
                    conf_label = "High Confidence"
                elif confidence > 0.55:
                    conf_label = "Moderate Confidence"
                else:
                    conf_label = "Toss-up"

                st.success(f"**Recommendation: {winner} to WIN** — {conf_label} ({confidence:.1%})")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ─────────────────────────────────────────────
# MODEL VS MARKET
# ─────────────────────────────────────────────
elif page == "💹 Model vs Market":
    st.title("💹 Model vs Market")
    st.markdown("Home-team win probability from the model and from Kalshi's game market "
                "(midpoint of the YES bid and ask). Each pre-tip-off prediction is saved to "
                "`data/prediction_log.csv`.")
    try:
        slate = get_slate()
    except Exception as e:
        slate = None
        st.error(f"Could not build today's slate: {e}")
    if slate is not None and slate.empty:
        st.info("No regular-season or playoff games today.")
    elif slate is not None:
        from daily_slate import slate_table
        st.dataframe(slate_table(slate), use_container_width=True, hide_index=True)
        st.caption("Highlighted rows: model and market differ by 5 points or more. Smaller gaps are within "
                   "Kalshi's fees and the model's normal error. On held-out 2025-26 games the market's "
                   "pre-tip-off price was more accurate than this model (see Model Performance), so a large "
                   "gap is more often the market knowing something (late scratches, rest) than an edge.")

# ─────────────────────────────────────────────
# TRACK RECORD
# ─────────────────────────────────────────────
elif page == "📈 Track Record":
    import prediction_log
    st.title("📈 Live Track Record")
    st.markdown("Predictions logged before tip-off, scored once results are ingested. "
                "For each game the last prediction made before tip-off counts.")
    record = prediction_log.track_record()
    if record.empty:
        st.info("No finished games with a logged prediction yet. Predictions are logged by "
                "`python src/daily_slate.py` (also run by the refresh scripts) and by the "
                "Model vs Market page; results arrive with `python src/ingest.py`.")
    else:
        from sklearn.metrics import log_loss, brier_score_loss
        rows = []
        for name, col in [('Model', 'MODEL_HOME_PROB'), ('Kalshi', 'MARKET_HOME_PROB')]:
            r = record.dropna(subset=[col])
            if len(r):
                p = r[col].clip(1e-6, 1 - 1e-6)
                rows.append({'Source': name, 'Games': len(r),
                             'Accuracy': ((p > 0.5) == r['HOME_WIN']).mean(),
                             'Log loss': log_loss(r['HOME_WIN'], p, labels=[0, 1]),
                             'Brier': brier_score_loss(r['HOME_WIN'], p)})
        st.dataframe(pd.DataFrame(rows).set_index('Source').style.format(
            {'Accuracy': '{:.1%}', 'Log loss': '{:.4f}', 'Brier': '{:.4f}'}))
        both = record.dropna(subset=['MARKET_HOME_PROB'])
        if len(both) < len(record):
            st.caption(f"Kalshi price available for {len(both)} of {len(record)} games.")

        st.subheader("Running log loss")
        fig, ax = plt.subplots(figsize=(10, 4))
        for name, col in [('Model', 'MODEL_HOME_PROB'), ('Kalshi', 'MARKET_HOME_PROB')]:
            m = prediction_log.running_metrics(record, col)
            if len(m):
                ax.plot(m['games'], m['log_loss'], label=name)
        ax.axhline(np.log(2), color='grey', linestyle=':', label='Coin flip (0.693)')
        ax.set_xlabel('Games')
        ax.set_ylabel('Cumulative log loss')
        ax.legend()
        st.pyplot(fig)
        plt.close()

        st.subheader("Calibration (live predictions)")
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        calib = record.assign(bucket=pd.cut(record['MODEL_HOME_PROB'], bins=bins, include_lowest=True))
        calib = calib.groupby('bucket', observed=True).agg(
            games=('HOME_WIN', 'size'), predicted=('MODEL_HOME_PROB', 'mean'), actual=('HOME_WIN', 'mean'))
        st.dataframe(calib.style.format({'predicted': '{:.1%}', 'actual': '{:.1%}'}))
        if len(record) < 100:
            st.caption(f"Only {len(record)} games so far: these numbers will move a lot until a few "
                       "hundred games are in.")

# ─────────────────────────────────────────────
# PAGE 2: BACKTEST RESULTS
# ─────────────────────────────────────────────
elif page == "📊 Backtest Results":
    st.title("📊 Backtest Results")
    st.markdown("Walk-forward held-out games only: each game was predicted by a model trained on earlier games. "
                "No betting simulation; the comparison with Kalshi's market prices is on the "
                "Model Performance page.")
    st.markdown("---")

    try:
        from sklearn.metrics import brier_score_loss, accuracy_score, log_loss

        preds = load_test_predictions()
        y_true = preds['TARGET']

        rows = []
        for name, col in [('Shipped model', 'MODEL_PROB'), ('Elo only', 'ELO_PROB')]:
            p = preds[col]
            rows.append({'Model': name, 'ROC-AUC': roc_auc_score(y_true, p),
                         'Accuracy': accuracy_score(y_true, p > 0.5),
                         'Log loss': log_loss(y_true, p), 'Brier': brier_score_loss(y_true, p)})
        st.dataframe(pd.DataFrame(rows).set_index('Model').style.format('{:.4f}'))
        st.caption(f"{len(preds)} games from {preds['GAME_DATE'].min():%b %d, %Y} "
                   f"to {preds['GAME_DATE'].max():%b %d, %Y}")

        st.subheader("Calibration")
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        calib = preds.assign(bucket=pd.cut(preds['MODEL_PROB'], bins=bins, include_lowest=True))
        calib = calib.groupby('bucket', observed=True).agg(
            games=('TARGET', 'size'), predicted=('MODEL_PROB', 'mean'), actual=('TARGET', 'mean'))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
        ax.plot(calib['predicted'], calib['actual'], 'o-', color='steelblue', label='Shipped model')
        ax.set_xlabel('Predicted home win probability')
        ax.set_ylabel('Actual home win rate')
        ax.legend()
        st.pyplot(fig)
        plt.close()

        st.subheader("Prediction Confidence Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(preds['MODEL_PROB'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0.5, color='red', linestyle='--', label='50% threshold')
        ax.set_xlabel('Predicted Win Probability')
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Backtest failed: {e}")

# ─────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("---")

    try:
        from sklearn.metrics import roc_curve

        metrics = pd.read_csv('data/test_metrics.csv')
        st.subheader("Walk-forward results (held-out games only)")
        st.dataframe(metrics.drop(columns=['shipped']).set_index('model').style.format('{:.4f}'))
        st.caption("Each test block is predicted by a model trained only on earlier games, with Elo "
                   "settings tuned on those training games. Lower log loss is better.")
        if os.path.exists('data/market_metrics.csv'):
            mm = pd.read_csv('data/market_metrics.csv')
            st.subheader(f"Against the market ({int(mm['games'].iloc[0])} held-out games, "
                         f"{mm['first_game'].iloc[0]} to {mm['last_game'].iloc[0]})")
            st.dataframe(mm[['model', 'roc_auc', 'accuracy', 'log_loss', 'brier']].set_index('model')
                         .style.format('{:.4f}'))
            st.caption("Kalshi's price at tip-off beats the model on these games.")

        test_preds = load_test_predictions()
        model = load_model()

        y_test = test_preds['TARGET']
        probs = test_preds['MODEL_PROB']
        preds = (probs > 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.4f}')
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Away Win', 'Home Win'])
            ax.set_yticklabels(['Away Win', 'Home Win'])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=16)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()

        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': feature_weights(model).values
        }).sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        ax.set_xlabel('Weight (logistic coefficient or tree importance)')
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Model performance failed: {e}")