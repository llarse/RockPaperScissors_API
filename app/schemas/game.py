from pydantic import BaseModel
from enum import Enum


class Choice(str, Enum):
    rock = "rock"
    paper = "paper"
    scissors = "scissors"


class GameRequest(BaseModel):
    user_choice: Choice
    last_choice: Choice = None


class GameResponse(BaseModel):
    user_choice: Choice
    ai_choice: Choice
    result: str
