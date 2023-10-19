import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def test_client():
    client = TestClient(app)
    yield client  # this is the value that will be used by the tests
