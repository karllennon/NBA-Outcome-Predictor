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
from nba_api.stats.static import teams as nba_teams_static
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

# Name mismatches between NBA API and our data
TEAM_NAME_MAP = {
    'Los Angeles Clippers': 'LA Clippers'
}

@st.cache_resource
def load_predictor():
    return GamePredictor()

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

def get_todays_games():
    try:
        from nba_api.stats.endpoints import scoreboardv2
        import time
        time.sleep(3)
        sb = scoreboardv2.ScoreboardV2()
        games = sb.get_data_frames()[0]
        if games.empty:
            return None
        all_teams = pd.DataFrame(nba_teams_static.get_teams())
        team_map = all_teams.set_index('id')['full_name'].to_dict()
        matchups = []
        for _, game in games.iterrows():
            home = TEAM_NAME_MAP.get(team_map.get(game['HOME_TEAM_ID']), team_map.get(game['HOME_TEAM_ID']))
            away = TEAM_NAME_MAP.get(team_map.get(game['VISITOR_TEAM_ID']), team_map.get(game['VISITOR_TEAM_ID']))
            if home and away:
                matchups.append((home, away))
        return matchups if matchups else None
    except Exception:
        return None

def run_prediction(home_team, away_team, home_injuries, away_injuries, home_acute, away_acute):
    result = load_predictor().predict(home_team, away_team, home_injuries, away_injuries,
                                      home_acute, away_acute)

    def describe(details):
        out = []
        for player, status, impact, boost in details:
            icon = "⚡" if status == 'acute' else "🔴"
            out.append(f"{icon} **{player}**: {status.upper()} (-{impact:.1f}, +{boost:.1f} boost)")
        return out

    return (result['home_prob'], describe(result['home_details']), describe(result['away_details']),
            result['injury_diff'], result['stale_warning'])

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/National_Basketball_Association_logo.svg/200px-National_Basketball_Association_logo.svg.png", width=80)
st.sidebar.title("NBA Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏀 Today's Slate", "📊 Backtest Results", "🤖 Model Performance"])
st.sidebar.markdown("---")
try:
    _state = pd.read_csv('data/team_state.csv', parse_dates=['LAST_GAME_DATE'])
    st.sidebar.caption(f"Data through: {_state['LAST_GAME_DATE'].max():%b %d, %Y}")
    _metrics = pd.read_csv('data/test_metrics.csv').set_index('model')
    _shipped = _metrics[_metrics['shipped']].index[0]
    st.sidebar.caption(f"Model: {_shipped} | Held-out AUC: {_metrics.loc[_shipped, 'roc_auc']:.3f} "
                       f"(Elo only: {_metrics.loc['Elo only (logistic)', 'roc_auc']:.3f})")
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

    todays_games = None

    # Disabled until NBA API issue is resolved
    # if 'todays_games' not in st.session_state:
    #     with st.spinner("Loading today's schedule..."):
    #         st.session_state.todays_games = get_todays_games()
    # todays_games = st.session_state.todays_games

    if todays_games:
        st.success(f"Found {len(todays_games)} games today")
        game_options = [f"{away} @ {home}" for home, away in todays_games]
        selected_game = st.selectbox("Select a game", game_options)
        idx = game_options.index(selected_game)
        home_team = todays_games[idx][0]
        away_team = todays_games[idx][1]
    else:
        st.info("Live schedule unavailable — select teams manually")
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("🏠 Home Team", all_teams, index=all_teams.index("Boston Celtics"))
        with col2:
            away_team = st.selectbox("✈️ Away Team", all_teams, index=all_teams.index("Los Angeles Lakers"))

    st.markdown("---")

    home_rotation = get_rotation(home_team)
    away_rotation = get_rotation(away_team)

    col1, col2 = st.columns(2)
    home_injuries = []
    home_acute = []
    away_injuries = []
    away_acute = []

    with col1:
        st.subheader(f"🏠 {home_team}")
        st.caption("✓ = OUT  |  New injury = last 1-3 games")
        if home_rotation:
            for i, player in enumerate(home_rotation):
                core_tag = "⭐ " if i < 4 else ""
                is_out = st.checkbox(
                    f"{core_tag}{player['PLAYER_NAME']} ({player['impact_score']:.1f})",
                    key=f"home_out_{home_team}_{i}"
                )
                if is_out:
                    home_injuries.append(player['PLAYER_NAME'])
                    is_new = st.checkbox(
                        f"   ⚡ New injury?",
                        key=f"home_acute_{home_team}_{i}"
                    )
                    if is_new:
                        home_acute.append(player['PLAYER_NAME'])
        else:
            st.warning("No rotation data found")

    with col2:
        st.subheader(f"✈️ {away_team}")
        st.caption("✓ = OUT  |  New injury = last 1-3 games")
        if away_rotation:
            for i, player in enumerate(away_rotation):
                core_tag = "⭐ " if i < 4 else ""
                is_out = st.checkbox(
                    f"{core_tag}{player['PLAYER_NAME']} ({player['impact_score']:.1f})",
                    key=f"away_out_{away_team}_{i}"
                )
                if is_out:
                    away_injuries.append(player['PLAYER_NAME'])
                    is_new = st.checkbox(
                        f"  ⚡ New injury?",
                        key=f"away_acute_{away_team}_{i}"
                    )
                    if is_new:
                        away_acute.append(player['PLAYER_NAME'])
        else:
            st.warning("No rotation data found")

    st.markdown("---")

    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing matchup..."):
            try:
                prob, home_details, away_details, injury_diff, stale_warning = run_prediction(
                    home_team, away_team, home_injuries, away_injuries, home_acute, away_acute
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
# PAGE 2: BACKTEST RESULTS
# ─────────────────────────────────────────────
elif page == "📊 Backtest Results":
    st.title("📊 Backtest Results")
    st.markdown("Walk-forward held-out games only: each game was predicted by a model trained on earlier games. "
                "No betting simulation, since the data has no historical odds.")
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