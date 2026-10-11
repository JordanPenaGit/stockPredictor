"""Versioned request cache. Exact intervals avoid silently stale overlapping downloads."""
import sqlite3, json
from datetime import datetime, timezone
import pandas as pd
class PriceCache:
    def __init__(self,path):
        self.path=path; path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, fetched TEXT NOT NULL, payload TEXT NOT NULL)')
    def connect(self): return sqlite3.connect(self.path,timeout=30)
    def get(self,key):
        with self.connect() as c: row=c.execute('SELECT fetched,payload FROM requests WHERE key=?',(key,)).fetchone()
        if not row:return None
        frame=pd.DataFrame(json.loads(row[1])); frame['date']=pd.to_datetime(frame['date'])
        frame=frame.set_index('date'); frame.attrs['fetched_at']=row[0]; return frame
    def put(self,key,frame):
        f=frame.reset_index(); f['date']=f['date'].dt.strftime('%Y-%m-%d')
        with self.connect() as c:c.execute('INSERT OR REPLACE INTO requests VALUES (?,?,?)',(key,datetime.now(timezone.utc).isoformat(),f.to_json(orient='records')))
