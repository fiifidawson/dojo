import pytest
from db import Database

@pytest.fixture
def db():
    """Provides a fresh instance of the Databases class and cleans up after the test."""
    database = Database()
    yield database