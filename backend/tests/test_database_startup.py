"""Starting up when the database can't be reached for a moment (seen live: a DNS blip)."""

import pytest
from sqlalchemy.exc import OperationalError

from app import storage
from app.storage import Database


class Flaky:
    """create_all that fails a given number of times, then works."""

    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def __call__(self, engine):
        self.calls += 1
        if self.calls <= self.failures:
            raise OperationalError("connect", {}, Exception("failed to resolve host 'db.example.com'"))


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(storage.time, "sleep", lambda s: None)


def test_a_blip_at_startup_is_retried(tmp_path, monkeypatch):
    flaky = Flaky(failures=2)
    monkeypatch.setattr(storage.metadata, "create_all", flaky)
    monkeypatch.setattr(Database, "_add_missing_columns", lambda self: None)
    Database(f"sqlite:///{tmp_path / 'x.db'}")
    assert flaky.calls == 3


def test_gives_up_with_a_plain_message(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.metadata, "create_all", Flaky(failures=99))
    with pytest.raises(RuntimeError, match="Check your internet connection and DATABASE_URL"):
        Database(f"sqlite:///{tmp_path / 'x.db'}")
