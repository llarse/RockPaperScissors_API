import pickle
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from app.schemas.game import Choice


class OneHiddenFF(nn.Module):
    def __init__(self, lr, input_dims, hidden_dim_1, hidden_dim_2, n_actions):
        super(OneHiddenFF, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.n_actions = n_actions

        # Functionally define sequental
        self.input_layer = nn.Linear(*self.input_dims, self.hidden_dim_1)
        self.hidden_layer = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.output_layer = nn.Linear(self.hidden_dim_1, self.n_actions)

        # Optimizer and Criterion
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # Use GPU if available
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        ''' Forward pass 
            Returns: probabilities of actions'''
        # Perform passes as sequental
        x = functional.relu(self.input_layer(state))
        x = functional.relu(self.hidden_layer(x))
        actions = self.output_layer(x)
        return actions

    def forwardInterface(self, state):
        ''' Interface function for forward pass
            Returns: singular action'''
        call = self.forward(torch.tensor(
            state, dtype=torch.float32).to(self.device))
        return torch.argmax(call).item()


class DQNAgent:
    def __init__(self, hidden_dims, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        # initialize memories with zeros
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        self.hidden_dims = hidden_dims
        self.Q_eval = OneHiddenFF(
            self.lr, self.input_dims, *hidden_dims, self.n_actions)

    def one_hot_encode(self, action):
        ''' one hot encodes move
        args: move = ROCK, PAPER, or SCISSORS'
        retuns: one_hot = [1,0,0] where the 1 is in the place of the move'''
        one_hot = np.zeros(3)
        move = 0
        if action == Choice.paper:
            move = 1
        elif action == Choice.scissors:
            move = 2
        one_hot[move] = 1
        return one_hot

    def store_transition(self, state, action, reward, next_state, terminal):
        ''' Stores the action, the consequence of that action,
            and the environement before and after the action'''
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        # Epsilon-greedy
        if np.random.random() > self.epsilon:
            # Exploit - take action based on a forward pass to the network
            state = torch.tensor(np.array(observation),
                                 dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            # Explore - take action based on a random choice
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        # reset the optimizer gradients
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        # Create a batch
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Send batch variables to device
        state_batch = torch.tensor(
            self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(
            self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(
            self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(
            self.terminal_memory[batch]).to(self.Q_eval.device)

        # calculate q values
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)[
            batch_index, action_batch]
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * \
            torch.max(q_next.view(-1, 1), dim=1)[0]

        # calculate loss
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
        return loss

    def store_agent(self, username):
        # Create a directory for user checkpoints if it doesn't exist
        user_dir = f"ROCKPAPERSCISSORSAPI/users/{username}"
        print(user_dir)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # Save agent's attributes
        agent_data = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "action_space": self.action_space,
            "mem_size": self.mem_size,
            "batch_size": self.batch_size,
            "n_actions": self.n_actions,
            "input_dims": self.input_dims,
            "mem_cntr": self.mem_cntr,
            "iter_cntr": self.iter_cntr,
            "replace_target": self.replace_target,
            "state_memory": self.state_memory,
            "new_state_memory": self.new_state_memory,
            "action_memory": self.action_memory,
            "reward_memory": self.reward_memory,
            "terminal_memory": self.terminal_memory
        }

        with open(os.path.join(user_dir, "agent.pkl"), "wb") as f:
            pickle.dump(agent_data, f)

        # Save Q_eval network state
        torch.save(self.Q_eval.state_dict(),
                   os.path.join(user_dir, "q_eval.pth"))

    def load_agent(self, username):
        user_dir = f"ROCKPAPERSCISSORSAPI/users/{username}"
        try:
            # Load agent's attributes
            with open(os.path.join(user_dir, "agent.pkl"), "rb") as f:
                agent_data = pickle.load(f)
        except FileNotFoundError:
            return False

        for key, value in agent_data.items():
            setattr(self, key, value)

        # Load Q_eval network state
        self.Q_eval.load_state_dict(torch.load(
            os.path.join(user_dir, "q_eval.pth")))
        self.Q_eval.eval()  # Set the network to evaluation mode
        return True
