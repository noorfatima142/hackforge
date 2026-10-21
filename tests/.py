import sqlite3
import pandas as pd
from src.status import STATUS_DB

con = sqlite3.connect(STATUS_DB)
df = pd.read_sql("SELECT * FROM quality_log ORDER BY log_id", con)
df.to_csv("quality_log.csv", index=False)
print(df.tail(15).to_string())