"""Supabase Storage file store, against a pretend Supabase that lives inside the test."""

import json

import httpx
import pandas as pd
import pytest

from app import session as session_module
from app.config import settings
from app.session import SessionManager
from app.storage import SupabaseFileStore, database_url


class FakeStorage:
    """Just enough of the Supabase Storage API: buckets, upload, download, delete."""

    def __init__(self):
        self.buckets, self.objects, self.uploads = set(), {}, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] == "sb_secret_test"
        path = request.url.path.removeprefix("/storage/v1")
        if path.startswith("/bucket"):
            if request.method == "GET":
                return httpx.Response(200 if path.split("/")[-1] in self.buckets else 404, json={})
            self.buckets.add(json.loads(request.content)["id"])
            return httpx.Response(200, json={})
        if request.method == "DELETE":
            for name in json.loads(request.content)["prefixes"]:
                self.objects.pop(name, None)
            return httpx.Response(200, json=[])
        name = path.split("/", 3)[3]  # /object/<bucket>/<name>
        if request.method == "POST":
            self.objects[name] = request.content
            self.uploads.append(name)
            return httpx.Response(200, json={})
        if name in self.objects:
            return httpx.Response(200, content=self.objects[name])
        return httpx.Response(400, json={"statusCode": "404", "error": "not_found", "message": "Object not found"})


@pytest.fixture
def cloud():
    return FakeStorage()


def store(cloud):
    return SupabaseFileStore("https://p.supabase.co", "sb_secret_test", "notebooks", transport=httpx.MockTransport(cloud))


def test_save_load_delete_and_missing(cloud):
    files = store(cloud)
    assert "notebooks" in cloud.buckets  # made on first use, private
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    files.save("s1_original", df)
    pd.testing.assert_frame_equal(files.load("s1_original"), df)
    files.delete("s1_original")
    assert files.load("s1_original") is None


def test_unchanged_tables_are_not_uploaded_again(cloud, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(session_module, "file_store", lambda: store(cloud))
    manager = SessionManager()
    manager.initialize()
    df = pd.DataFrame({"x": range(40), "y": [i % 2 for i in range(40)]})
    session = manager.create_session(df, "d.csv")
    assert cloud.uploads == ["%s_original.parquet" % session.session_id]

    for _ in range(5):  # e.g. cells autosaving
        manager.save(session)
    assert len(cloud.uploads) == 1

    session.target_column = "y"
    session.lock_test_rows(0.25)
    manager.save(session)
    assert sorted(cloud.uploads[1:]) == sorted(f"{session.session_id}_{k}.parquet" for k in ("train", "test"))

    # A fresh server rebuilds the notebook from the cloud, then has nothing new to upload
    other = SessionManager()
    other.initialize()
    reloaded = other.get_session(session.session_id)
    assert len(reloaded.test_df) == len(session.test_df)
    other.save(reloaded)
    assert len(cloud.uploads) == 3


def test_supabase_urls_use_the_psycopg_driver(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://postgres.abc:pw@aws-0.pooler.supabase.com:5432/postgres")
    assert database_url().startswith("postgresql+psycopg://postgres.abc:pw@")
