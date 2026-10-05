from inference import GamePredictor

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = GamePredictor()
        report = _predictor.load_injury_report()
        if report is not None:
            print(f"Injury report: {report['REPORT_TIME'].max()} ({len(report)} rows)")
        elif _predictor.report_error:
            print(f"[!] Injury report unavailable: {_predictor.report_error}")
    return _predictor


def _print_rotation(team_name, rotation, out_players):
    print(f"\n[ROTATION: {team_name}]")
    for i, p in rotation.iterrows():
        tag = " <- CORE" if i < 4 else ""
        out_tag = " (OUT)" if p['PLAYER_NAME'] in out_players else ""
        print(f"  {i + 1}. {p['PLAYER_NAME']} (impact: {p['impact_score']:.1f}){tag}{out_tag}")


def predict_game(home_team_name, away_team_name, home_injuries=None, away_injuries=None,
                 game_date=None):
    """
    home_injuries / away_injuries: players out for this game; None = use the latest
        official injury report (players listed Out)
    """
    result = get_predictor().predict(home_team_name, away_team_name,
                                     home_injuries, away_injuries, game_date=game_date)
    home_injuries, away_injuries = result['home_out'], result['away_out']
    for side, source in result['out_sources'].items():
        print(f"  {side} injuries: {'from injury report' if source == 'report' else 'no report for this game'}")

    print("\n" + "=" * 50)
    print(f" MATCHUP: {away_team_name} @ {home_team_name}")
    print("=" * 50)
    if result['stale_warning']:
        print(f"\n[!] {result['stale_warning']}")

    _print_rotation(home_team_name, result['home_rotation'], home_injuries)
    _print_rotation(away_team_name, result['away_rotation'], away_injuries)

    print("\n[INJURY ANALYSIS]")
    print(f"  Injury differential: {result['injury_diff']:+.2f} (positive favors {home_team_name})")
    for team, details in [(home_team_name, result['home_details']), (away_team_name, result['away_details'])]:
        for player, status, impact in details:
            print(f"  {team}: {player} {status.upper()} (-{impact:.1f})")
    for player in result['skipped']:
        print(f"  [!] {player} is not in the current top-10 rotation, so not counted")

    prob = result['home_prob']
    pick = home_team_name if prob > 0.5 else away_team_name
    print("\n[PREDICTION]")
    print(f"  {home_team_name} win probability: {prob:.1%}")
    print(f"  Pick: {pick}")
    print("=" * 50 + "\n")
    return result


def _ask_list(prompt, empty=None):
    text = input(prompt).strip()
    if text.lower() == 'none':
        return []
    return [p.strip() for p in text.split(',')] if text else empty


if __name__ == "__main__":
    h_team = input("Enter Home Team Name: ")
    a_team = input("Enter Away Team Name: ")

    prompt = "(comma separated; Enter = use injury report; 'none' = nobody out): "
    home_out = _ask_list(f"\nInjured players for {h_team} {prompt}")
    away_out = _ask_list(f"\nInjured players for {a_team} {prompt}")

    predict_game(h_team, a_team, home_out, away_out)
