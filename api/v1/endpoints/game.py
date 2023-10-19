from fastapi import APIRouter
from app.schemas.game import GameRequest, GameResponse
from app.services.game_service import get_ai_choice, determine_winner

router = APIRouter()


@router.post("/", response_model=GameResponse)
def play_game(request: GameRequest):
    user_choice = request.user_choice
    ai_choice = get_ai_choice()
    result = determine_winner(user_choice, ai_choice)
    return {"user_choice": user_choice, "ai_choice": ai_choice, "result": result}
