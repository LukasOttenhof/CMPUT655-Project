import gymnasium as gym
from stable_baselines3 import SAC
from tbu_gym.tbu_continous import TruckBackerEnv_C
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
# Code Taken Basically From https://stable-baselines3.readthedocs.io/en/master/modules/sac.html

total_timesteps = 200_000

checkpoint_callback = CheckpointCallback(
save_freq=10_000,
save_path="./logs/",
name_prefix="rl_model",
save_replay_buffer=True,
save_vecnormalize=False,
)

env = TruckBackerEnv_C(seed=0)

model = SAC("MlpPolicy", env, verbose=1,  learning_starts=30_000, n_steps=7, ent_coef="auto") 
model.learn(total_timesteps=total_timesteps, log_interval=4, callback=checkpoint_callback)
model.save("another_tuned_sac_tbu_continous") # File Path for Saved Agent 

del model # remove to demonstrate saving and loading

model = SAC.load("./expert_policy_files/tuned_sac_tbu_continous")

# Running Agent 
print("Running Loaded Agent for 10K Steps")
demo_steps = 0
env = TruckBackerEnv_C(seed=0)
obs, info = env.reset()
while demo_steps <= 10_000:
    demo_steps += 1
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
