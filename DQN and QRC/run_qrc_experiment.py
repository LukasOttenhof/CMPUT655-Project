from tbu_gym.tbu_discrete import TruckBackerEnv_D
from agents import QRCAgent, DQNAgent
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from experiment import Experiment, QRC_AGENT

# --- Environment ---
env = TruckBackerEnv_D(render_mode=None)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

num_episodes = 10000
max_steps_per_episode = 5000
gamma = 0.99
learning_rate = 1e-3
epsilon_start = 1.0
epsilon_decay = 0.999997
epsilon_min = 0.01
batch_size = 64
target_update_freq = 5  # Update target net every few episodes

qrc_expirement = Experiment(agent_name=QRC_AGENT)
# qrc_expirement.run_single_visual()
qrc_expirement.run_multiple_visual()