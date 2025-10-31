# importing the environment 
from tbu_jax.tbu_discrete import TBUax_d
# other imports
import jax
from jax import numpy as jnp
from typing import (Any,Generic,TypeVar,overload)
from functools import partial

@partial(jax.jit, static_argnums=(2,))
def step_env(accum, unused, env):
    # unpacking
    env_state, env_params, total_return, rng = accum
    # getting random action
    rng, sample_rng = jax.random.split(rng)
    action = env.action_space().sample(sample_rng)
    # stepping the environment 
    rng, rng_step = jax.random.split(rng)
    obs, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
    total_return += reward
    return (env_state, env_params, total_return, rng), 0.0

# Making Gymnax Version 
env = TBUax_d()
env_params = env.default_params

# Running 100k of the random uniform policy via lax scan, since it's jax
key = jax.random.PRNGKey(0)
key, key_reset = jax.random.split(key)
obs_0, state_0 = env.reset(key_reset, env_params)
results_of_running, _ = jax.lax.scan(partial(step_env,env=env), init=(state_0, env_params, 0.0, key), 
    xs=None, length=100_000)
print("100k steps yielded a performance of :", results_of_running[-2])