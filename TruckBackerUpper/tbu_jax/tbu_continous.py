"""JAX version of Truck Backer Upper environment."""
from typing import Any, Dict, Optional, Tuple, Union, NamedTuple
import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
)
from functools import partial


# # Uncomment Below for Debugging
#jax.config.update("jax_disable_jit", True)

TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")

@struct.dataclass
class EnvState(environment.EnvState):
    # Visible State Variables
    x: jnp.ndarray
    y: jnp.ndarray
    theta_c: jnp.ndarray
    theta_t: jnp.ndarray
    # Hidden State Variables
    time: int

class Reset_Bounds(NamedTuple):
    x : tuple
    y : tuple
    theta_t : tuple
    theta_c : tuple 


@struct.dataclass
class EnvParams(environment.EnvParams):
    l_t: float = 14.0
    l_c: float = 6.0
    x_bounds: tuple =  (0,200)
    y_bounds: tuple = (-100, 100) 
    dist_tol : float = 5.0
    angle_tol : float = 0.5
    jack_tol : float = jnp.pi/2
    max_angle : float = 4*jnp.pi
    max_steps_in_episode: int = 300  
    restart_bounds : Reset_Bounds = Reset_Bounds(x=(100,150), y=(-20,20), theta_t=(-1,1), theta_c=(-0.5, 0.5))


class TBUax_c(environment.Environment[EnvState, EnvParams]):

    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def at_goal(self, state: EnvState, params : EnvParams):
        at_goal = jnp.logical_and(jnp.sqrt((state.x**2 + state.y**2) <= params.dist_tol), jnp.abs(state.theta_t) <= params.angle_tol)
        return at_goal
    
    def is_jackknifed(self, state: EnvState, params : EnvParams):
        jack_knifed = state.theta_c > params.jack_tol
        return jack_knifed

    def valid_location(self, state : EnvState, params : EnvParams):
        valid_x = jnp.logical_and(state.x >= params.x_bounds[0], state.x <= params.x_bounds[1])
        valid_y = jnp.logical_and(state.y >= params.y_bounds[0], state.y <= params.y_bounds[1])
        valid_loc = jnp.logical_and(valid_x, valid_y)
        return valid_loc

    def valid_angles(self, state : EnvState, params : EnvParams):
        return state.theta_t <= params.max_angle


    def step_env(
        self,
        key: chex.PRNGKey, 
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Intermediate Variables for computation
        a = 3.0 * jnp.cos(action)
        b = a * jnp.cos(state.theta_c)
        # Updating State Variables
        x_new = state.x - b * jnp.cos(state.theta_t)
        y_new = state.y - b * jnp.sin(state.theta_t)
        theta_t_new = state.theta_t - jnp.arcsin(a * jnp.sin(state.theta_c) / params.l_t)
        theta_c_new = state.theta_c + jnp.arcsin(3.0 * jnp.sin(action) / (params.l_c + params.l_t))
        # Updating State Dictionary
        new_state = EnvState(
            x=x_new.squeeze(),
            y=y_new.squeeze(),
            theta_t=theta_t_new.squeeze(),
            theta_c=theta_c_new.squeeze(),
            time=state.time + 1,)
        # Checking for Possible Termination Conditions
        is_jacked = self.is_jackknifed(new_state, params)
        is_valid_loc = self.valid_location(new_state, params)
        is_valid_angle = self.valid_angles(new_state, params)
        # Computing Termination Condition
        terminated_goal = self.at_goal(new_state, params)
        valid_loc_and_angles = jnp.logical_and(is_valid_loc, is_valid_angle)
        valid = jnp.logical_and(jnp.logical_not(is_jacked), valid_loc_and_angles)
        terminated_fail = jnp.logical_not(valid)
        # Computing Reward
        reward = 101*terminated_goal + -1 
        # Resetting the Truck but not the Environment
        new_state = jax.lax.cond(terminated_fail, lambda x : self.reset_truck(x, new_state, params), lambda x : new_state, key)
        # Computing Done 
        done = jnp.logical_or(terminated_goal, (new_state.time == 300))
        # Returning things in the Gymnax Style
        return (
            lax.stop_gradient(self.get_obs(new_state, params)),
            lax.stop_gradient(new_state),
            reward,
            done, {"discount": self.discount(new_state, params)}) 

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key_one, key_two, key_three, key_four = jax.random.split(key, 4)
        state = EnvState(
            x= jax.random.uniform(key_one, minval=params.restart_bounds.x[0], maxval=params.restart_bounds.x[1], shape=(1,)).squeeze(),
            y = jax.random.uniform(key_two, minval=params.restart_bounds.y[0], maxval=params.restart_bounds.y[1], shape=(1,)).squeeze(), 
            theta_c = jax.random.uniform(key_three, minval=params.restart_bounds.theta_c[0], maxval=params.restart_bounds.theta_c[1], shape=(1,)).squeeze(),
            theta_t = jax.random.uniform(key_four, minval=params.restart_bounds.theta_t[0], maxval=params.restart_bounds.theta_t[1], shape=(1,)).squeeze(),
            time = 0,
        )
        return self.get_obs(state, params), state
    

    def reset_truck(
        self, key: chex.PRNGKey, state : EnvState, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key_one, key_two, key_three, key_four = jax.random.split(key, 4)
        state = EnvState(
            x= jax.random.uniform(key_one, minval=params.restart_bounds.x[0], maxval=params.restart_bounds.x[1], shape=(1,)).squeeze(),
            y = jax.random.uniform(key_two, minval=params.restart_bounds.y[0], maxval=params.restart_bounds.y[1], shape=(1,)).squeeze(), 
            theta_c = jax.random.uniform(key_three, minval=params.restart_bounds.theta_c[0], maxval=params.restart_bounds.theta_c[1], shape=(1,)).squeeze(),
            theta_t = jax.random.uniform(key_four, minval=params.restart_bounds.theta_t[0], maxval=params.restart_bounds.theta_t[1], shape=(1,)).squeeze(),
            time = state.time,
        )
        return state

    def get_obs(self, state: EnvState, params : EnvParams, key=None) -> chex.Array:
        """Applies observation function to state."""
        # We include state-scaling here and remove the time variable 
        normed_x = (6*state.x / params.x_bounds[1] - 3)
        normed_y = (3*state.y / params.y_bounds[1])
        obs = jnp.array([normed_x,normed_y, state.theta_c, state.theta_t])
        x = jnp.reshape(obs, (-1,))
        return x

    # Below are the standard Gymnax functions

    @property
    def name(self) -> str:
        """Environment name."""
        return "TBUax_c"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-1.0, 1.0, shape=(1,), dtype=jnp.float)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low=jnp.array([-3, -3, -2*jnp.pi, -2*jnp.pi], dtype=float)
        high=jnp.array([3, 3, 2*jnp.pi, 2*jnp.pi], dtype=float)
        return spaces.Box(low, high, shape=(4,), dtype=float)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "x": spaces.Box(params.x_bounds[0], params.x_bounds[1], (), jnp.float),
                "y": spaces.Box(params.y_bounds[0], params.y_bounds[1], (), jnp.float),
                "theta_t": spaces.Box(-2*jnp.pi, 2*jnp.pi, (), jnp.float),
                "theta_c": spaces.Box(-2*jnp.pi, 2*jnp.pi, (), jnp.float),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state transition is terminal."""
        # Computing Termination Condition
        terminated_goal = self.at_goal(state, params)
        done = jnp.logical_or(terminated_goal, (state.time == 300))
        return done
# This is a way to avoid placing this in the gymnax registry, and having to edit your local gymnax 
# there's certaintly a better way to accomplish this.
   
    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams | None = None,
    ) -> tuple[jax.Array, TEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: TEnvParams | None = None
    ) -> tuple[jax.Array, TEnvState]:
        """Performs resetting of environment."""
        if params is None:
            params = self.default_params

        # Reset
        obs, state = self.reset_env(key, params)

        return obs, state   

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__