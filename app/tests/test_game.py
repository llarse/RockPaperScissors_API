from fastapi.testclient import TestClient
from app.main import app
from app.schemas.game import Choice
from tests.conftest import test_client

client = test_client(app)


def test_play_game():
    response = client.post("/v1/game/", json={"user_choice": Choice.rock})
    assert response.status_code == 200
    data = response.json()
    assert "user_choice" in data
    assert "ai_choice" in data
    assert "result" in data
    assert data["user_choice"] == Choice.rock.value
