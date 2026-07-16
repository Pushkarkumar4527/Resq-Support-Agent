import sqlite3
import pandas as pd

conn = sqlite3.connect('tickets.db')
df = pd.read_sql_query("SELECT * FROM support_tickets", conn)
print("\n--- Current SQLite Database Entries ---")
print(df)
conn.close()