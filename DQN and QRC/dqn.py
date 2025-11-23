import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from tqdm import tqdm
from rl_glue import RLGlue
from tbu_gym.tbu_discrete import TruckBackerEnv_D
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from tbu_gym.tbu_discrete import TruckBackerEnv_D
from tqdm import tqdm  # for live progress bar


# Hyperparameters
num_episodes = 1000
max_steps_per_episode = 500
gamma = 0.99
learning_rate = 1e-3
epsilon_start = 1.0
epsilon_decay = 0.99997
epsilon_min = 0.01
batch_size = 64
target_update_freq = 5

# for one session, this will be used by threads
def run_single_experiment(seed=None):
    if seed is not None:
        np.random.seed(seed)

    env = TruckBackerEnv_D(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.99997,
        epsilon_min=0.01,
        batch_size=64
    )

    episode_rewards = []
    for episode in range(1, 1001):
        state = env.reset()
        total_reward = 0

        for t in range(500):
            action = agent.agent_policy(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            agent.train_with_mem()
            state = next_state
            if done:
                break

        if episode % 5 == 0:
            agent.update_target()

        episode_rewards.append(total_reward)

    return np.array(episode_rewards)


# --- Main execution guard (MUST HAVE ON WINDOWS) ---
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # <- recommended for Windows

    num_runs = 200
    num_threads = 8
    all_results = []

    print(f"Running {num_runs} experiments using {num_threads} threads")

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_single_experiment, seed=i) for i in range(num_runs)]
        for f in tqdm(as_completed(futures), total=num_runs, desc="Training progress", ncols=100):
            result = f.result()
            all_results.append(result)

    all_results = np.array(all_results)

    mean_rewards = np.mean(all_results, axis=0)
    ci_low = np.percentile(all_results, 10, axis=0)
    ci_high = np.percentile(all_results, 90, axis=0)
    episodes = np.arange(1, 1001)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=episodes, y=mean_rewards, label='Mean Reward')
    plt.fill_between(episodes, ci_low, ci_high, alpha=0.3, label='95% CI')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training on TruckBackerEnv_D (200 runs)')
    plt.legend()
    plt.show()