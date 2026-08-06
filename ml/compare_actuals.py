import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://oracle:oracle@localhost:5432/oracle')

query = '''
    SELECT 
        p.full_name, p.team, pr.week, pr.stat,
        pr.projection as projected_stat,
        CASE
            WHEN pr.stat = 'passing_yards' THEN fm.actual_passing_yards
            WHEN pr.stat = 'passing_tds' THEN fm.actual_passing_tds
            WHEN pr.stat = 'rushing_yards' THEN fm.actual_rushing_yards
            WHEN pr.stat = 'rushing_tds' THEN fm.actual_rushing_tds
            WHEN pr.stat = 'receiving_yards' THEN fm.actual_receiving_yards
            WHEN pr.stat = 'receiving_tds' THEN fm.actual_receiving_tds
            ELSE NULL
        END as actual_stat
    FROM projections pr
    JOIN players p ON pr.player_id = p.id
    JOIN feature_matrix fm ON fm.player_id = pr.player_id AND fm.season = pr.season AND fm.week = pr.week
    WHERE pr.season = 2024 AND p.position = 'QB' 
      AND pr.stat IN ('passing_yards', 'passing_tds', 'rushing_yards')
    ORDER BY p.full_name, pr.week DESC
    LIMIT 15;
'''

try:
    df = pd.read_sql(query, engine)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df)
except Exception as e:
    print('Error:', e)

