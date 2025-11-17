import numpy as np
from tqdm import tqdm

from rl_glue import RLGlue

import torch
import torch.nn as nn
import torch.optim as optim
import random

from collections import deque

class QRCAgent:
    """
    Q-learning with Regularized Correction (QRC)
    ----------------------------------------------------
    - Q network learns TD error + correction from h(s,a)
    - h network learns to approximate TD error (delta)
    - h has L2 regularization (weight_decay = beta)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=50000,
        batch_size=64,
        beta=1,        # regularization on h-net (weight decay) => 1 # 1e-5
        device=None,
    ):
        # Basic configs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.beta = beta

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Device
        self.device = (
            device
            or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Networks
        self.q_nn = self.build_nn(output_dim=action_dim).to(self.device)
        self.target_net = self.build_nn(output_dim=action_dim).to(self.device)
        self.h_net = self.build_nn(output_dim=action_dim).to(self.device)

        # Initialize target net
        self.target_net.load_state_dict(self.q_nn.state_dict())
        self.target_net.eval()

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_nn.parameters(), lr=self.lr)
        self.h_optimizer = optim.AdamW(
            self.h_net.parameters(),
            lr=self.lr,
            weight_decay=self.beta * self.lr,    # ← REGULARIZATION on h
        )

        self.steps = 0

    def build_nn(self, output_dim):
        """Simple 2-layer MLP"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def agent_policy(self, state):
        """Epsilon-greedy"""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_nn(state)
        return int(q_values.argmax(dim=1).item())


    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))


    def compute_loss(self, batch):
        """Compute v_loss (Q-update) and h_loss (correction learner) and print RC info"""

        # Unpack batch
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device) 
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q(s,a)
        q_values = self.q_nn(states)
        # q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_sa = q_values.gather(1, actions)

        #TODO : replace target_net with q_nn if we can't use it

        # Target Q
        # with torch.no_grad():
        #     next_q = self.target_net(next_states).max(1)[0]
        #     target = rewards + self.gamma * (1 - dones) * next_q

        next_q = self.target_net(next_states).max(1)[0]
        target = rewards + self.gamma * (1 - dones) * next_q

        # TD ERROR δ
        delta = q_sa - target.detach()

        # Predict correction h(s,a)
        h_values = self.h_net(states)
        h_sa = h_values.gather(1, actions).squeeze(1)

        # Correction term => Don't detach next_q here
        # correction_term = self.gamma * h_sa.detach() * next_q.detach()
        correction_term = self.gamma * h_sa.detach() * next_q

        # QRC losses
        v_loss = 0.5 * delta.pow(2) + correction_term
        h_loss = 0.5 * (delta.detach() - h_sa).pow(2)

        return v_loss.mean(), h_loss.mean()

    def train_with_mem(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Compute QRC losses
        v_loss, h_loss = self.compute_loss(
            (states, actions, rewards, next_states, dones)
        )

        # Optimize Q-net
        self.q_optimizer.zero_grad()
        v_loss.backward()
        self.q_optimizer.step()

        # Optimize h-net
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Target network update
    def update_target(self):
        self.target_net.load_state_dict(self.q_nn.state_dict())


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-5, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, buffer_size=50000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        # device, not needed, but needed if going on canada compute to specify gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_nn = self.build_nn().to(self.device) # build q network
        self.target_net = self.build_nn().to(self.device) # build target network
        self.target_net.load_state_dict(self.q_nn.state_dict()) # make target net same as q net. initialization is random so need this
        self.optimizer = optim.Adam(self.q_nn.parameters(), lr=lr) 
        
    def build_nn(self): # to build the q network, 2 hiddne layer with relu and 128 neuronsin each
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
    
    def agent_policy(self, state): # act e greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim) # rand action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # convert state to tensor, then add batch dimension, then move to device
        with torch.no_grad(): # dont calculate gradients, no need
            q_values = self.q_nn(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # store transition sars, done is if terminal state
    
    def train_with_mem(self): # train with experience from memory using batch size set in agent
        if len(self.memory) < self.batch_size: # if not enought mem, could be changed to use what we have instead of skip
            return
        batch = random.sample(self.memory, self.batch_size) # get batch
        states, actions, rewards, next_states, dones = zip(*batch) # get batch features

        # convert data to tensors which can be used by pytorch
        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device) 
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # get current q values from q network
        q_values = self.q_nn(states).gather(1, actions)

        # use the samples from experience batch to calc target network q values
        with torch.no_grad(): # no need to calc gradients for target
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]  
        # target = r + gamma * max_a' Q(s', a') * (1 - done)
        target = rewards + self.gamma * next_q_values * (1 - dones)

        # mean squared error 
        errors = target - q_values
        squared_errors = errors ** 2
        mean_squared_error = torch.mean(squared_errors)

        self.optimizer.zero_grad()
        mean_squared_error.backward()
        self.optimizer.step()
 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 
    
    def update_target(self): # update target network replacing it with the current q network
        self.target_net.load_state_dict(self.q_nn.state_dict())