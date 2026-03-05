import streamlit as st
import pandas as pd
import joblib
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from features import NBAFeatureProcessor
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

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed_data_with_elo.csv')
    player_data = pd.read_csv('data/raw_player_boxscores.csv')
    player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE'])
    player_data.columns = player_data.columns.str.strip()
    return df, player_data

@st.cache_data
def load_training_set():
    df = pd.read_csv('data/final_training_set.csv')
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

@st.cache_resource
def load_model():
    return joblib.load('models/nba_model.joblib')

@st.cache_data
def load_positions():
    return pd.read_csv('data/player_positions.csv')

@st.cache_data
def get_all_teams():
    all_teams = nba_teams_static.get_teams()
    names = [t['full_name'] for t in all_teams]
    names = [TEAM_NAME_MAP.get(n, n) for n in names]
    return sorted(names)

@st.cache_data
def get_all_rotations():
    df, player_data = load_data()
    processor = NBAFeatureProcessor(df)
    rotation_df = processor.identify_core_four(player_data).reset_index()
    if 'TEAM_NAME' not in rotation_df.columns:
        teams_df = pd.DataFrame(nba_teams_static.get_teams())[['id', 'full_name']]
        teams_df.columns = ['TEAM_ID', 'TEAM_NAME']
        rotation_df = rotation_df.merge(teams_df, on='TEAM_ID', how='left')
    # Apply name map
    rotation_df['TEAM_NAME'] = rotation_df['TEAM_NAME'].replace(TEAM_NAME_MAP)
    return rotation_df.groupby('TEAM_NAME').apply(
        lambda x: x[['PLAYER_NAME', 'impact_score']].to_dict('records')
    ).to_dict()

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
            home = team_map.get(game['HOME_TEAM_ID'])
            away = team_map.get(game['VISITOR_TEAM_ID'])
            if home and away:
                matchups.append((home, away))
        return matchups if matchups else None
    except Exception:
        return None

def run_prediction(home_team, away_team, home_injuries, away_injuries, home_acute, away_acute):
    # Apply name map so stats lookup works
    home_team_data = TEAM_NAME_MAP.get(home_team, home_team)
    away_team_data = TEAM_NAME_MAP.get(away_team, away_team)

    df, player_data = load_data()
    model = load_model()
    positions_df = load_positions()
    processor = NBAFeatureProcessor(df)

    rotation_dict = get_all_rotations()
    max_boost_tracker = {}

    def calc_impact_loss(team_name, injured_players, acute_overrides):
        total_lost = 0
        total_boost = 0
        details = []

        for pname in injured_players:
            if pname in acute_overrides:
                injury_type = 'acute'
            else:
                injury_type = processor.classify_injury(pname, player_data)

            team_rot = rotation_dict.get(team_name, [])
            player_row = next((p for p in team_rot if p['PLAYER_NAME'] == pname), None)
            if player_row is None:
                continue

            impact = player_row['impact_score']
            total_lost += impact

            if injury_type == 'acute':
                boost, _ = processor.calculate_replacement_boost(
                    pname, team_name, player_data, positions_df, max_boost_tracker
                )
                total_boost += boost
                details.append(f"⚡ **{pname}** — ACUTE (-{impact:.1f}, +{boost:.1f} boost)")
            else:
                details.append(f"🔴 **{pname}** — CHRONIC (-{impact:.1f}, rotation adjusted)")

        return total_lost - total_boost, details

    home_loss, home_details = calc_impact_loss(home_team, home_injuries, home_acute)
    away_loss, away_details = calc_impact_loss(away_team, away_injuries, away_acute)
    injury_diff = (away_loss - home_loss) / 10

    h_data = df[df['TEAM_NAME'] == home_team_data]
    a_data = df[df['TEAM_NAME'] == away_team_data]
    home_stats = h_data.iloc[-1]
    away_stats = a_data.iloc[-1]

    features = pd.DataFrame([{
        'ELO_DIFF': home_stats['PRE_GAME_ELO'] - away_stats['PRE_GAME_ELO'],
        'EFG_DIFF': home_stats['ROLLING_eFG_PCT'] - away_stats['ROLLING_eFG_PCT'],
        'TOV_PCT_DIFF': home_stats['ROLLING_TOV_PCT'] - away_stats['ROLLING_TOV_PCT'],
        'ORB_PCT_DIFF': home_stats['ROLLING_ORB_PCT'] - away_stats['ROLLING_ORB_PCT'],
        'FT_RATE_DIFF': home_stats['ROLLING_FT_RATE'] - away_stats['ROLLING_FT_RATE'],
        'WIN_STREAK_DIFF': home_stats['WIN_STREAK'] - away_stats['WIN_STREAK'],
        'REST_DIFF': home_stats['DAYS_REST'] - away_stats['DAYS_REST'],
        'B2B_DIFF': home_stats['IS_B2B'] - away_stats['IS_B2B'],
        'PLUS_MINUS_DIFF': home_stats['ROLLING_PLUS_MINUS'] - away_stats['ROLLING_PLUS_MINUS'],
        'PACE_DIFF': home_stats['ROLLING_PACE'] - away_stats['ROLLING_PACE'],
        'DEF_RATING_DIFF': home_stats['ROLLING_DEF_RATING'] - away_stats['ROLLING_DEF_RATING'],
        'CORE_INJURY_DIFF': injury_diff
    }])

    prob = model.predict_proba(features)[0][1]
    return prob, home_details, away_details, injury_diff

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/National_Basketball_Association_logo.svg/200px-National_Basketball_Association_logo.svg.png", width=80)
st.sidebar.title("NBA Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏀 Today's Slate", "📊 Backtest Results", "🤖 Model Performance"])
st.sidebar.markdown("---")
st.sidebar.caption("Data through: Feb 11, 2026")
st.sidebar.caption("Model: XGBoost | ROC-AUC: 0.66")

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

    rotation_dict = get_all_rotations()
    home_rotation = rotation_dict.get(home_team, [])
    away_rotation = rotation_dict.get(away_team, [])

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
                core_tag = "⭐ " if i < 5 else ""
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
                core_tag = "⭐ " if i < 5 else ""
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
                prob, home_details, away_details, injury_diff = run_prediction(
                    home_team, away_team, home_injuries, away_injuries, home_acute, away_acute
                )
                away_prob = 1 - prob

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
    st.markdown("Flat $10 betting simulation on all games in training set")
    st.markdown("---")

    try:
        from sklearn.metrics import brier_score_loss, accuracy_score

        df_bt = load_training_set()
        model = load_model()

        X = df_bt.drop(columns=['TARGET'])
        y_true = df_bt['TARGET']
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        results = pd.DataFrame({
            'Actual': y_true.values,
            'Predicted': preds,
            'Probability': probs
        })

        def calc_profit(row):
            if row['Predicted'] == 1:
                return 9.10 if row['Actual'] == 1 else -10.0
            return 0

        results['Profit'] = results.apply(calc_profit, axis=1)
        results['Cumulative'] = results['Profit'].cumsum()

        accuracy = accuracy_score(y_true, preds)
        brier = brier_score_loss(y_true, probs)
        roc = roc_auc_score(y_true, probs)
        total_profit = results['Profit'].sum()
        bets_placed = (results['Predicted'] == 1).sum()
        total_invested = bets_placed * 10
        roi = total_profit / total_invested if total_invested > 0 else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{accuracy:.1%}")
        m2.metric("ROC-AUC", f"{roc:.4f}")
        m3.metric("Brier Score", f"{brier:.4f}")
        m4.metric("Total P&L", f"${total_profit:.2f}", delta="profit" if total_profit > 0 else "loss")
        m5.metric("ROI", f"{roi:.1%}")

        st.markdown("---")
        st.subheader("Cumulative P&L Over Time")
        st.line_chart(results['Cumulative'])

        st.subheader("Prediction Confidence Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(probs, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
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
        from sklearn.model_selection import train_test_split

        df_train = load_training_set()
        model = load_model()

        X = df_train.drop(columns=['TARGET'])
        y = df_train['TARGET']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

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
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score')
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Model performance failed: {e}")