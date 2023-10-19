from fastapi.testclient import TestClient
from app.main import app
from tests.conftest import test_client

client = test_client(app)


def test_health_check():
    response = client.get("/v1/health-check/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
