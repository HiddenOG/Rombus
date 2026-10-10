"""Copy the notebooks on this machine (SQLite + Parquet files in DATA_DIR) to the cloud:
the DATABASE_URL database and Supabase Storage. The local copies are left untouched.

Safe to run twice: records and files are replaced, never duplicated.

    python -m app.migrate
"""

import os
import sys

from app.config import settings
from app.storage import Database, LocalFileStore, SupabaseFileStore, database_url, file_store


def main() -> int:
    if not settings.DATABASE_URL or not settings.SUPABASE_SECRET_KEY:
        print("Set DATABASE_URL and SUPABASE_SECRET_KEY in backend/.env first.")
        return 1
    local_path = os.path.join(settings.DATA_DIR, "sessions.db")
    if not os.path.exists(local_path):
        print("No local notebooks to move.")
        return 0

    local_db, local_files = Database(f"sqlite:///{local_path}"), LocalFileStore(settings.DATA_DIR)
    cloud_db, cloud_files = Database(database_url()), file_store()
    assert isinstance(cloud_files, SupabaseFileStore)

    ids = [r["session_id"] for r in local_db.recent(limit=1_000_000)]
    print(f"Moving {len(ids)} notebook(s) to Supabase...")
    for sid in ids:
        row = local_db.get(sid)
        for suffix in ("_original", "_train", "_test"):
            df = local_files.load(f"{sid}{suffix}")
            if df is not None:
                cloud_files.save(f"{sid}{suffix}", df)
        cloud_db.save(row)
        print(f"  {row['filename']} ({sid[:8]})")
    print("Done. Local copies are unchanged; delete backend/data yourself once you're happy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
