from fastapi import APIRouter, Depends
import logging

from app.schemas.game import GameRequest, GameResponse, Choice
from app.services.game_service import get_ai_choice, determine_winner
from app.services.dqn_agent import DQNAgent
from app.api.dependencies import get_verified_jwt

router = APIRouter()

GAMMA = 0.99  # Discount factor
EPSILON = 1.0  # Starting Epsilon
BATCH_SIZE = 64
EPS_DEC = 5e-4  # Epsilon decay rate
EPS_END = 0.001  # Ending epsioln
INPUT_DIMS = 3
LR = 0.001  # learning rate

HIDDEN_DIMS = [64, 64]

agent = DQNAgent(HIDDEN_DIMS, gamma=GAMMA, epsilon=EPSILON, batch_size=BATCH_SIZE,
                 n_actions=3, eps_dec=EPS_DEC, eps_end=EPS_END, input_dims=[INPUT_DIMS], lr=LR)


@router.post("/start_game")
def start_game(username: str = Depends(get_verified_jwt)):
    global agent
    catch = agent.load_agent(username)
    if catch:
        return {"status": "Game started - Loaded checkpoint"}
    return {"status": "Game started - No checkpoint found"}


@router.post("/end_game")
def end_game(username: str = Depends(get_verified_jwt)):
    agent.store_agent(username)  # Store the agent's state


@router.post("/", response_model=GameResponse)
def play_game(request: GameRequest):
    # set current and last choice
    user_choice = request.user_choice

    last_choice = request.last_choice
    if request.last_choice is None:
        # Use rock if there is no last choice (simulate aggressive player)
        last_choice = Choice.rock

    # Encode the choice
    user_state = agent.one_hot_encode(user_choice)
    last_state = agent.one_hot_encode(last_choice)

    # get the Agent's action using the last state
    ai_state = agent.choose_action(last_state)
    ai_choice = Choice.rock if ai_state == 0 else Choice.paper if ai_state == 1 else Choice.scissors

    # get the winner of the game
    result = determine_winner(user_choice, ai_choice)
    logging.info(f"Game result: {result}")
    # determine the reward for the AI
    reward = -1 if result == "user" else 1
    if result == "draw":
        reward = 0

    # Store the transition in state
    # always set terminal to True because RPS is a 1 round game
    agent.store_transition(last_state, ai_state, reward, user_state, True)

    # Train the agent
    loss = agent.learn()

    # Return the game result
    # the user choice should be used again as the "last choice"

    return {"user_choice": user_choice, "ai_choice": ai_choice, "result": result}
