import functools
import jax
import jax.numpy as jnp
from gymnasium import spaces
from pettingzoo import ParallelEnv


class PongEnvironment(ParallelEnv):
    name = "2PlayerPongParallell"
    input_size = 6
    output_size = 3

    def __init__(self, num_envs=1024, max_bounces=100, max_steps=1000):
        self.timestep_reward = 1 / max_steps
        self.num_envs = num_envs
        self.max_bounces = max_bounces
        self.max_steps = max_steps
        self.width = 800
        self.height = 400
        self.aspect_ratio = self.width / self.height

        # State variables as batched arrays
        self.timestep = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self.pong_x = jnp.full((self.num_envs,), 0.5)
        self.pong_y = jnp.full((self.num_envs,), 0.5)
        self.pong_vx = jnp.zeros(self.num_envs)
        self.pong_vy = jnp.zeros(self.num_envs)
        self.left_paddle_y = jnp.full((self.num_envs,), 0.5)
        self.right_paddle_y = jnp.full((self.num_envs,), 0.5)
        self.pong_speed = jnp.full((self.num_envs,), 0.02)
        self.paddle_speed = 0.01
        self.paddle_length = 0.25 / 2
        self.pong_speed_increase = 1.03
        self.max_pong_speed = 1.0

    def observe(self):
        return {
            0: jnp.stack([self.pong_x, self.pong_y, self.pong_vx, self.pong_vy, self.left_paddle_y,
                          self.right_paddle_y], axis=1),
            1: jnp.stack(
                [1 - self.pong_x, self.pong_y, -self.pong_vx, self.pong_vy, self.right_paddle_y,
                 self.left_paddle_y], axis=1),
        }

    def reset(self, seed=None, options=None):
        random_key = jax.random.PRNGKey(seed or 0)
        self.timestep = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self.pong_speed = jnp.full((self.num_envs,), 0.02)
        self.pong_x = jnp.full((self.num_envs,), 0.5)
        self.pong_y = jnp.full((self.num_envs,), 0.5)

        start_rad = jax.random.uniform(random_key, (self.num_envs,), minval=-jnp.pi / 16,
                                       maxval=jnp.pi / 16)
        start_side = jax.random.choice(random_key, jnp.array([-1, 1]), shape=(self.num_envs,))

        self.pong_vx = start_side * self.pong_speed * jnp.cos(start_rad)
        self.pong_vy = self.pong_speed * jnp.sin(start_rad)

        self.left_paddle_y = jnp.full((self.num_envs,), 0.5)
        self.right_paddle_y = jnp.full((self.num_envs,), 0.5)

        return self.observe(), {}

    def step(self, actions):
        left_paddle_action = actions[0]
        right_paddle_action = actions[1]

        # Update paddle positions (vectorized)
        self.left_paddle_y = jnp.clip(
            self.left_paddle_y + (left_paddle_action == 1) * self.paddle_speed - (
                        left_paddle_action == 2) * self.paddle_speed,
            0.0, 1.0)

        self.right_paddle_y = jnp.clip(
            self.right_paddle_y + (right_paddle_action == 1) * self.paddle_speed - (
                        right_paddle_action == 2) * self.paddle_speed,
            0.0, 1.0)

        # Update ball positions (vectorized)
        self.pong_x += self.pong_vx / self.aspect_ratio
        self.pong_y += self.pong_vy

        # Handle ball collisions with top and bottom walls (vectorized)
        condition = self.pong_y < 0
        self.pong_y = jnp.where(condition, -self.pong_y, self.pong_y)
        self.pong_vy = jnp.where(condition, -self.pong_vy, self.pong_vy)

        condition = self.pong_y > 1
        self.pong_y = jnp.where(condition, 2 - self.pong_y, self.pong_y)
        self.pong_vy = jnp.where(condition, -self.pong_vy, self.pong_vy)

        # Rewards and termination (vectorized)
        rewards = jnp.full((self.num_envs, 2), 0.0)
        done = self.timestep >= self.max_steps

        # Handle ball collisions with paddles (left and right, vectorized)
        left_hit = (self.pong_x < 0) & (self.left_paddle_y - self.paddle_length < self.pong_y) & (
                    self.pong_y < self.left_paddle_y + self.paddle_length)
        right_hit = (self.pong_x > 1) & (self.right_paddle_y - self.paddle_length < self.pong_y) & (
                    self.pong_y < self.right_paddle_y + self.paddle_length)

        # Update ball velocity and position on paddle hit
        self.pong_speed = jnp.where(left_hit | right_hit, jnp.minimum(self.max_pong_speed,
                                                                      self.pong_speed * self.pong_speed_increase),
                                    self.pong_speed)
        self.pong_vx = jnp.where(left_hit, -self.pong_vx, self.pong_vx)
        self.pong_vx = jnp.where(right_hit, -self.pong_vx, self.pong_vx)

        # Update timestep
        self.timestep += 1

        return self.observe(), rewards, done, 

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0]),
                          high=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=jnp.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(3)


if __name__ == "__main__":
    env = PongEnvironment()
    # Testing and running the environment would need adaptation to support JAX operations
