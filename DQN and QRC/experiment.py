import matplotlib.pyplot as plt
import numpy as np

class Experiment:
    def __init__(
        self,
        env,
        agent,
        num_episodes=500,
        max_steps_per_episode=200,
        target_update_freq=10,
    ):
        """
        env: OpenAI Gym / Gymnasium compatible environment
        agent: either DQNAgent or QRCAgent instance
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode
        self.target_update_freq = target_update_freq

        self.episode_rewards = []
        self.recent_loss = 0.0

    def run(self):
        for episode in range(1, self.num_episodes + 1):
            reset_output = self.env.reset()
            if isinstance(reset_output, tuple):
                state, _ = reset_output  # Gymnasium-style reset
            else:
                state = reset_output     # Classic Gym

            total_reward = 0

            for t in range(self.max_steps):
                # --- Select action ---
                action = self.agent.agent_policy(state)

                # --- Step environment ---
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result

                total_reward += reward

                # --- Store transition & train ---
                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.train_with_mem()
                if loss != 0.0:
                    self.recent_loss = loss

                state = next_state
                if done:
                    break

            # --- Update target network periodically ---
            if episode % self.target_update_freq == 0:
                self.agent.update_target()

            # --- Logging ---
            self.episode_rewards.append(total_reward)
            # print(f"Episode {episode:4d}/{self.num_episodes}, "
            #       f"Reward: {total_reward:7.2f}, "
            #       f"Epsilon: {self.agent.epsilon:.5f}, "
            #       f"Loss: {self.recent_loss:.5f}")
            print(f"Episode {episode:4d}/{self.num_episodes}, "
                  f"Reward: {total_reward:7.2f}, "
                  f"Epsilon: {self.agent.epsilon:.5f}")

    def plot_results(self, title="Training Rewards", save_path=None, smooth_window=50):
        plt.figure(figsize=(14, 6))
        
        # Plot raw rewards
        plt.plot(self.episode_rewards, label="Episode Reward", alpha=0.4)

        # --- Smoothing ---
        if len(self.episode_rewards) >= smooth_window:
            window = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(self.episode_rewards, window, mode='valid')
            x_smooth = np.arange(smooth_window - 1, len(self.episode_rewards))
            plt.plot(x_smooth, smoothed, label=f"Smoothed (window={smooth_window})", linewidth=2.0)

        # --- Plot setup ---
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def run_visual(self):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(12, 6))
        line_raw, = ax.plot([], [], label="Episode Reward", alpha=0.4)
        line_smooth, = ax.plot([], [], label="Smoothed Reward", linewidth=2.0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Training Progress (Live)")
        ax.legend()
        ax.grid(True)

        smooth_window = 50
        rewards = []

        for episode in range(1, self.num_episodes + 1):
            reset_output = self.env.reset()
            state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
            total_reward = 0

            for t in range(self.max_steps):
                action = self.agent.agent_policy(state)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result

                total_reward += reward
                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.train_with_mem()
                if loss != 0.0:
                    self.recent_loss = loss
                state = next_state
                if done:
                    break

            # --- Update target network periodically ---
            if episode % self.target_update_freq == 0:
                self.agent.update_target()

            # --- Logging ---
            rewards.append(total_reward)
            self.episode_rewards = rewards
            self.recent_loss = getattr(self, "recent_loss", 0.0)

            # --- Update live plot ---
            line_raw.set_data(np.arange(len(rewards)), rewards)
            if len(rewards) >= smooth_window:
                window = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(rewards, window, mode='valid')
                x_smooth = np.arange(smooth_window - 1, len(rewards))
                line_smooth.set_data(x_smooth, smoothed)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)

            print(f"Episode {episode:4d}/{self.num_episodes}, "
                f"Reward: {total_reward:7.2f}, "
                f"Epsilon: {self.agent.epsilon:.5f}")

        plt.ioff()  # Turn off interactive mode
        plt.show()

