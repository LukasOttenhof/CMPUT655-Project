import numpy as np
from tqdm import tqdm

from rl_glue import RLGlue

import torch
import torch.nn as nn
import torch.optim as optim
import random

from collections import deque

class QRCAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        lam=0.9,                # kept for API compatibility (not used here)
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        buffer_size=50000,
        batch_size=64,
        beta=1.0,               # regularization strength for h (matches JAX self.beta)
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.beta = beta

        # Replay memory
        self.memory = deque(maxlen=buffer_size)

        # Device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Networks
        self.q_net = self.build_nn(output_dim=self.action_dim).to(self.device)
        self.target_net = self.build_nn(output_dim=self.action_dim).to(self.device)
        self.h_net = self.build_nn(output_dim=self.action_dim).to(self.device)

        # init target
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.h_optimizer = optim.Adam(self.h_net.parameters(), lr=self.lr)

        # Steps counter
        self.steps = 0

    def build_nn(self, output_dim):
        """2 hidden layers with 128 neurons each"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def agent_policy(self, state):
        """Epsilon-greedy policy returning an int action"""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        """Store transition tuple"""
        self.memory.append((state, action, reward, next_state, done))

    def train_with_mem(self):
        """Train using experience replay. Returns total loss (q_loss + h_loss) for logging."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)         # (B, state_dim)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)     # (B,1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)  # (B,1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)      # (B,1)

        # Current Q for taken actions (B,1)
        q_all = self.q_net(states)                                  # (B, A)
        q_values = q_all.gather(1, actions)                         # (B,1)

        # h-values for taken actions (B,1)
        h_all = self.h_net(states)
        h_values = h_all.gather(1, actions)                         # (B,1)

        # Next-state greedy value (use target_net), detach it
        with torch.no_grad():
            next_q_all = self.target_net(next_states)               # (B,A)
            # greedy next action value: max_a Q'(s', a)
            vtp1 = next_q_all.max(dim=1, keepdim=True)[0]          # (B,1)
            vtp1 = vtp1.detach()

            # Q target (standard Q-learning greedy target)
            target = rewards + self.gamma * vtp1 * (1.0 - dones)   # (B,1)
            target = target.detach()

        # TD error (not detached yet)
        delta = target - q_values                                   # (B,1)

        # Build v_loss and h_loss per-sample exactly like JAX:
        # v_loss_i = 0.5 * delta^2 + gamma * stop_gradient(delta_hat) * vtp1
        # h_loss_i = 0.5 * (stop_gradient(delta) - delta_hat)^2
        v_loss_terms = 0.5 * (delta ** 2) + self.gamma * h_values.detach() * vtp1
        h_loss_terms = 0.5 * (delta.detach() - h_values) ** 2

        v_loss = v_loss_terms.mean()
        h_loss = h_loss_terms.mean()

        # --- Update Q-network (q_net) ---
        self.q_optimizer.zero_grad()
        v_loss.backward(retain_graph=True)   # retain_graph because h_loss backward next might need graph if shared (here nets separate so optional)
        self.q_optimizer.step()

        # --- Update h-network (h_net) ---
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        # --- Apply explicit decay / regularization on h parameters (beta term) ---
        # This approximates: params_h <- params_h - stepsize * beta * params_h
        # i.e. multiplicative shrinkage: p *= (1 - lr * beta)
        if self.beta != 0.0:
            decay_factor = 1.0 - (self.lr * self.beta)
            # clamp decay_factor to a sensible positive number
            decay_factor = max(decay_factor, 0.0)
            with torch.no_grad():
                for p in self.h_net.parameters():
                    p.mul_(decay_factor)

        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon

        total_loss = v_loss.item() + h_loss.item()
        return total_loss

    def update_target(self):
        """Hard update target network weights from q_net"""
        self.target_net.load_state_dict(self.q_net.state_dict())


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