from tbu_gym.tbu_discrete import TruckBackerEnv_D
from agents import QRCAgent, DQNAgent
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# --- Environment ---
env = TruckBackerEnv_D(render_mode=None)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

num_episodes = 1000
max_steps_per_episode = 500
gamma = 0.99
learning_rate = 1e-3
epsilon_start = 1.0
epsilon_decay = 0.99997
epsilon_min = 0.01
batch_size = 64
target_update_freq = 5  # Update target net every few episodes

# --- Agent: choose either QRCAgent or DQNAgent ---
# agent = QRCAgent(
#     state_dim=state_dim,
#     action_dim=action_dim,
#     lr=learning_rate,
#     gamma=gamma,
#     epsilon=epsilon_start,
#     epsilon_decay=epsilon_decay,
#     min_epsilon=epsilon_min,
#     batch_size=batch_size
# )

# --- Run Experiment ---
# experiment = Experiment(env, agent, num_episodes=1000, max_steps_per_episode=500)
# experiment.run()
# experiment.plot_results(title="QRC Training on TruckBackerEnv_D",
#                         save_path="QRC_Discrete_TruckBackerEnv_D.png")

# experiment.run_visual()

# experiment.plot_results(title="QRC Training on TruckBackerEnv_D",
#                         save_path="results/QRC_Discrete_TruckBackerEnv.png")

# experiment = Experiment(env, agent, num_episodes=1000, max_steps_per_episode=500)
# experiment.run_multiple(agent, num_runs=20, save_path="results/QRC_Result.png")

all_rewards = []
num_runs = 50

for run in range(num_runs):
    # set random seeds for reproducibility
    print(f"Run {run+1}/{num_runs} Started.")
    seed = run
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # reinitialize environment and agent each run
    env = TruckBackerEnv_D(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = QRCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=learning_rate,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        batch_size=batch_size
    )

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0

        for t in range(max_steps_per_episode):
            action = agent.agent_policy(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            agent.train_with_mem()
            state = next_state
            if done:
                break

        if episode % target_update_freq == 0:
            agent.update_target()

        episode_rewards.append(total_reward)

    all_rewards.append(episode_rewards)
    print(f"Run {run+1}/{num_runs} finished.")

# --- Compute mean and std over runs ---
mean_rewards_dqp = np.mean(all_rewards, axis=0)
std_rewards_dqp = np.std(all_rewards, axis=0)

# --- Plot learning curve with error bands ---
plt.plot(mean_rewards_dqp, label="Mean Reward")
plt.fill_between(range(num_episodes),
                 mean_rewards_dqp - std_rewards_dqp,
                 mean_rewards_dqp + std_rewards_dqp,
                 alpha=0.2)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.title(f"QRC ({num_runs} runs)")
plt.show()