import random
from app.schemas.game import Choice


def get_ai_choice() -> Choice:
    return random.choice(list(Choice))


def determine_winner(user_choice: Choice, ai_choice: Choice) -> str:
    if user_choice == ai_choice:
        return "draw"
    if (
        (user_choice == Choice.rock and ai_choice == Choice.scissors) or
        (user_choice == Choice.scissors and ai_choice == Choice.paper) or
        (user_choice == Choice.paper and ai_choice == Choice.rock)
    ):
        return "user"
    return "ai"
