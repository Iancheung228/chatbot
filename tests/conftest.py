"""
Shared test fixtures.

temp_db: patches api.db.DB_PATH to a throwaway SQLite file in tmp_path
so tests never touch the real chatbot.db.
"""
import pytest
from unittest.mock import patch


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test.db")
    with patch("api.db.DB_PATH", db_file):
        from api.db import init_db
        init_db()
        yield db_file
