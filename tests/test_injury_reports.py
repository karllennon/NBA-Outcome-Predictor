import os
import pandas as pd
import pytest
import injury_reports as ir

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'Injury-Report_2026-03-03_05_00PM.pdf')


@pytest.mark.parametrize('report_name, box_name', [
    ('Doncic, Luka', 'Luka Dončić'),
    ('Butler, Jimmy', 'Jimmy Butler III'),
    ('Bagley III, Marvin', 'Marvin Bagley III'),
    ('Jackson Jr., Jaren', 'Jaren Jackson Jr.'),
    ('Washington, P.J.', 'P.J. Washington'),
    ('Gilgeous-Alexander, Shai', 'Shai Gilgeous-Alexander'),
    ("Melton, De'Anthony", "De'Anthony Melton"),
    ('Krejci, Vit', 'Vít Krejčí'),
])
def test_report_names_match_box_scores(report_name, box_name):
    matcher = ir.NameMatcher({'Team': [box_name, 'Someone Else']})
    assert matcher.match(report_name, 'Team') == box_name


def test_match_falls_back_to_last_name_and_initial():
    matcher = ir.NameMatcher({'Brooklyn Nets': ['Nic Claxton', 'Cam Thomas']})
    assert matcher.match('Claxton, Nicolas', 'Brooklyn Nets') == 'Nic Claxton'


def test_match_is_team_scoped_and_rejects_ambiguity():
    matcher = ir.NameMatcher({'A': ['Jalen Williams', 'Jaylin Williams'], 'B': ['Luka Dončić']})
    assert matcher.match('Doncic, Luka', 'A') is None
    assert matcher.match('Williams, J.', 'A') is None


def test_team_key_ignores_spacing():
    assert ir.team_key('DallasMavericks') == ir.team_key('Dallas Mavericks')


def test_url_formats():
    assert ir.report_url(pd.Timestamp('2024-01-15 17:30')).endswith('Injury-Report_2024-01-15_05PM.pdf')
    assert ir.report_url(pd.Timestamp('2026-03-03 17:15')).endswith('Injury-Report_2026-03-03_05_15PM.pdf')


def test_slots_are_at_or_before_time():
    new = ir.report_slots_before(pd.Timestamp('2026-03-03 17:10'), max_hours=1)
    assert new[0] == pd.Timestamp('2026-03-03 17:00') and all(s <= pd.Timestamp('2026-03-03 17:10') for s in new)
    old = ir.report_slots_before(pd.Timestamp('2024-01-15 18:00'), max_hours=3)
    assert old == [pd.Timestamp('2024-01-15 17:30'), pd.Timestamp('2024-01-15 16:30'),
                   pd.Timestamp('2024-01-15 15:30')]


def test_tip_time_parsing():
    assert ir.parse_tip_time('2026-03-03', '07:30(ET)') == pd.Timestamp('2026-03-03 19:30')
    assert ir.parse_tip_time('2026-03-03', '12:00(ET)') == pd.Timestamp('2026-03-03 12:00')
    assert ir.parse_tip_time('2026-03-03', '01:00(ET)') == pd.Timestamp('2026-03-03 13:00')
    assert ir.parse_tip_time('2026-03-03', '11:00(ET)') == pd.Timestamp('2026-03-03 23:00')  # PHX@SAC


def test_parse_fixture_report():
    with open(FIXTURE, 'rb') as f:
        df = ir.parse_report(f.read())
    assert list(df.columns) == ir.COLUMNS
    assert len(df) == 112
    assert df['REPORT_TIME'].iloc[0] == pd.Timestamp('2026-03-03 17:00')
    assert set(df['STATUS']) <= set(ir.STATUSES)
    assert not df['TEAM_NAME'].str.fullmatch(r'\d+').any()  # page footers not read as teams
    bagley = df[df['PLAYER'] == 'Marvin Bagley III'].iloc[0]
    assert bagley['TEAM_NAME'] == 'Dallas Mavericks' and bagley['STATUS'] == 'Out'
    assert bagley['REASON'] == 'Injury/Illness-Neck; Sprain'
    # reason wrapped over two lines in the PDF
    marshall = df[df['PLAYER'] == 'Naji Marshall'].iloc[0]
    assert marshall['REASON'] == 'Injury/Illness-Right Finger; Contusion'


def test_last_report_before_tip_ignores_later_reports():
    rows = []
    for rt, status in [('2026-03-03 17:00', 'Questionable'), ('2026-03-03 18:45', 'Out'),
                       ('2026-03-03 19:15', 'Available')]:
        rows.append({'REPORT_TIME': pd.Timestamp(rt), 'GAME_DATE': pd.Timestamp('2026-03-03'),
                     'GAME_TIME': '07:30(ET)', 'TEAM_NAME': 'Dallas Mavericks',
                     'PLAYER_NAME': 'Flagg, Cooper', 'STATUS': status})
    reports = pd.DataFrame(rows)
    got = ir.last_report_before_tip(reports, '2026-03-03', 'Dallas Mavericks')
    # tip 19:30, cutoff 19:00 -> the 18:45 report; the 19:15 one is too late
    assert got['STATUS'].tolist() == ['Out']
