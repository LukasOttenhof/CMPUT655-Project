import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from agents import QRCAgent, DQNAgent
from tbu_gym.tbu_discrete import TruckBackerEnv_D

QRC_AGENT = "QRC_AGENT"
DQN_Agent = "DQN_Agent"

class Experiment:
    def __init__(
        self,
        num_episodes=1000,
        max_steps_per_episode=500,
        gamma = 0.99,
        learning_rate = 1e-3,
        epsilon_start = 1.0,
        epsilon_decay = 0.99997,
        epsilon_min = 0.01,
        batch_size = 64,
        target_update_freq=10 ,
        agent_name = QRC_AGENT
    ):
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.agent_name = agent_name
        self.episode_rewards = []

    def run_single(self):
        env = TruckBackerEnv_D(render_mode=None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        episode_rewards = []

        if self.agent_name == QRC_AGENT:
            agent = QRCAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=self.learning_rate,
                gamma=self.gamma,
                epsilon=self.epsilon_start,
                epsilon_decay=self.epsilon_decay,
                epsilon_min=self.epsilon_min,
                batch_size=self.batch_size
            )
        elif self.agent_name == DQN_Agent:
            agent = DQNAgent(
                state_dim = state_dim,
                action_dim = action_dim,
                lr = self.learning_rate,
                gamma = self.gamma,
                epsilon = self.epsilon_start,
                epsilon_decay = self.epsilon_decay,
                epsilon_min = self.epsilon_min,
                batch_size = self.batch_size
            )
        else:
            return episode_rewards
        

        for episode in range(1, self.num_episodes + 1):
            reset_output = env.reset()
            if isinstance(reset_output, tuple):
                state, _ = reset_output  # Gymnasium-style reset
            else:
                state = reset_output     # Classic Gym

            total_reward = 0

            for t in range(self.max_steps_per_episode):
                # --- Select action ---
                action = agent.agent_policy(state)

                # --- Step environment ---
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result

                total_reward += reward

                # --- Store transition & train ---
                agent.remember(state, action, reward, next_state, done)
                loss = agent.train_with_mem()
                if loss != 0.0:
                    self.recent_loss = loss

                state = next_state
                if done:
                    break

            # --- Update target network periodically ---
            if episode % self.target_update_freq == 0:
                agent.update_target()

            # --- Logging ---
            episode_rewards.append(total_reward)
            if self.recent_loss:
                print(f"Episode {episode:4d}/{self.num_episodes}, "
                    f"Reward: {total_reward:7.2f}, "
                    f"Epsilon: {agent.epsilon:.5f}, "
                    f"Loss: {self.recent_loss:.5f}")
            else:
                print(f"Episode {episode:4d}/{self.num_episodes}, "
                    f"Reward: {total_reward:7.2f}, "
                    f"Epsilon: {agent.epsilon:.5f}")
        return episode_rewards
    
    def run_single_visual(self, save_path=None, smooth_window=50):
        episode_rewards = self.run_single()
            
        plt.figure(figsize=(14, 6))
        
        # Plot raw rewards
        plt.plot(episode_rewards, label="Episode Reward", alpha=0.4)

        # --- Smoothing ---
        if len(episode_rewards) >= smooth_window:
            window = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(episode_rewards, window, mode='valid')
            x_smooth = np.arange(smooth_window - 1, len(episode_rewards))
            plt.plot(x_smooth, smoothed, label=f"Smoothed (window={smooth_window})", linewidth=2.0)

        # --- Plot setup ---
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f"Traning {self.agent_name} Reward")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return episode_rewards
    
    def run_multiple(self, num_runs=5):
        """
        Run multiple experiments and return average reward per episode.
        Returns:
            avg_rewards (np.array): Mean reward per episode across runs.
            all_rewards (list): List of reward lists from each run.
        """
        all_rewards = []

        for run in range(1, num_runs + 1):
            print(f"\n===== Running Experiment {run}/{num_runs} =====")
            rewards = self.run_single()
            all_rewards.append(rewards)

        # Convert to array: shape -> (num_runs, num_episodes)
        all_rewards = np.array(all_rewards)

        # Average across runs
        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        print("\n===== Multi-Run Complete =====")
        return avg_rewards, std_rewards, all_rewards
    
    def run_multiple_visual(self, num_runs=5, smooth_window=50):
        """
        Run multiple experiments and plot average + std deviation band.
        """
        avg_rewards, std_rewards, all_rewards = self.run_multiple(num_runs)

        plt.figure(figsize=(20, 20))

        # Plot mean reward
        plt.plot(avg_rewards, label="Average Reward", linewidth=2)

        # Confidence band (±1 std)
        lower = avg_rewards - std_rewards
        upper = avg_rewards + std_rewards
        plt.fill_between(
            range(len(avg_rewards)),
            lower, upper,
            alpha=0.2, label="±1 Std Dev"
        )

        # Optional smoothing
        if len(avg_rewards) >= smooth_window:
            window = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(avg_rewards, window, mode="valid")
            plt.plot(smoothed, label=f"Smoothed Avg (window={smooth_window})")

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f"Average Reward Over {num_runs} Runs — {self.agent_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

        return avg_rewards, std_rewards, all_rewards

    

