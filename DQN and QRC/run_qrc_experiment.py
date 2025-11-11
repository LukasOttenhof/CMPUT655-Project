from tbu_gym.tbu_discrete import TruckBackerEnv_D
from agents import QRCAgent, DQNAgent
from experiment import Experiment

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
agent = QRCAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=learning_rate,
    gamma=gamma,
    epsilon=epsilon_start,
    epsilon_decay=epsilon_decay,
    min_epsilon=epsilon_min,
    batch_size=batch_size
)

# --- Run Experiment ---
experiment = Experiment(env, agent, num_episodes=1000, max_steps_per_episode=500)
experiment.run()
# experiment.plot_results(title="QRC Training on TruckBackerEnv_D",
#                         save_path="QRC_Discrete_TruckBackerEnv_D.png")

# experiment.run_visual()

experiment.plot_results(title="QRC Training on TruckBackerEnv_D",
                        save_path="result/QRC_Discrete_TruckBackerEnv.png")