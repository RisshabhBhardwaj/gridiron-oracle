import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from pipeline.normalize import Normalizer

def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set")
        sys.exit(1)

    print("Connecting to DB to check current snap count pop...")
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT season, COUNT(*) FROM game_logs WHERE offense_snaps IS NOT NULL GROUP BY season ORDER BY season")
            rows = cur.fetchall()
            print("Before normalize:")
            for r in rows:
                print(f"  Season {r[0]}: {r[1]} rows with snaps")

    print("\nRunning normalizer for all unprocessed rows...")
    with Normalizer(db_url) as norm:
        summary = norm.run(seasons=list(range(2019, 2026)))
        summary.log()

    print("\nConnecting to DB to check new snap count pop...")
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT season, COUNT(*) FROM game_logs WHERE offense_snaps IS NOT NULL GROUP BY season ORDER BY season")
            rows = cur.fetchall()
            print("After normalize:")
            for r in rows:
                print(f"  Season {r[0]}: {r[1]} rows with snaps")

if __name__ == "__main__":
    main()
