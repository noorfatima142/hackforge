import sqlite3
import pandas as pd
from src.status import STATUS_DB, _conn

_conn().close()   # makes sure the tables exist

con = sqlite3.connect(STATUS_DB)
df = pd.read_sql("SELECT * FROM quality_log ORDER BY log_id", con)
con.close()

if df.empty:
    print("quality_log is empty: run the agent on the real data first (Step 3)")
else:
    df.to_csv("quality_log.csv", index=False)
    print(df.tail(15).to_string())