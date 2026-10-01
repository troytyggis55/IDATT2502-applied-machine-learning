import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy


class VolleyballEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],
                "name": "volleyball_v0", "render_fps": 50
                }

    def __init__(self, render_mode=None, player_radius=0.05, ball_radius=0.05, player_speed=0.01,
                 ball_speed=0.015, gravity=4e-4, window_height=400, opponent=None, max_steps=200):
        # Ball X, Ball Y, Ball X Velocity, Ball Y Velocity, Left Player X, Left Player Y, Right Player X, Right Player Y, Time
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -ball_speed, -ball_speed, 0.0, 0.0, 1.0, 0.0, 0.0]),
            high=np.array([2.0, 1.0, ball_speed, ball_speed, 1.0, 1.0, 2.0, 1.0, 1.0]),
            dtype=np.float64)

        self.action_space = spaces.Discrete(5)  # Only 5 actions: nothing, left, right, up, down

        self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([-player_speed, 0]),
            2: np.array([player_speed, 0]),
            3: np.array([0, -player_speed]),
            4: np.array([0, player_speed])
        }
        self._mirror_action = {
            0: 0,
            1: 2,
            2: 1,
            3: 3,
            4: 4
        }

        self.player_radius = player_radius
        self.ball_radius = ball_radius
        self.ball_speed = ball_speed
        self.gravity = np.array([0, gravity])
        self.collision_dist = player_radius + ball_radius

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.window_size = window_height
        self.clock = None

        self.ball_pos = None
        self.ball_vel = None
        self.left_pos = None
        self.left_vel = None
        self.right_pos = None
        self.right_vel = None
        self.last_touch = None
        self.entered_right = None

        self.steps = None
        self.max_steps = max_steps

        self.opponent: BasePolicy | None = opponent


    def update_opponent(self, state_dict):
        if self.opponent is not None:
            self.opponent.load_state_dict(state_dict)
        else:
            print("Cannot update opponent as opponent is not set")

    def _get_obs(self):
        return np.array([self.ball_pos[0], self.ball_pos[1], self.ball_vel[0], self.ball_vel[1],
                         self.left_pos[0], self.left_pos[1], self.right_pos[0], self.right_pos[1],
                         np.float64(self.steps / self.max_steps)])
    def get_obs_mirror(self):
        return np.array(
            [2 - self.ball_pos[0], self.ball_pos[1], -self.ball_vel[0], self.ball_vel[1],
             2 - self.right_pos[0], self.right_pos[1], 2 - self.left_pos[0], self.left_pos[1],
             np.float64(self.steps / self.max_steps)])

    def _get_info(self):
        return {"touch": self.last_touch}

    def reset(self, seed=None, options=None):
        self.steps = 0

        super().reset(seed=seed)

        start_rad = self.np_random.uniform(-np.pi / 4, 0)  # - pi/8 to 0
        start_side = self.np_random.choice([-1, 1])

        self.ball_pos = np.array([1, 0.25])
        self.ball_vel = np.array([start_side * self.ball_speed * np.cos(start_rad),
                                  self.ball_speed * np.sin(start_rad)])

        self.left_pos = np.array([0.5, 0.5])
        self.left_vel = np.array([0, 0])
        self.right_pos = np.array([1.5, 0.5])
        self.right_vel = np.array([0, 0])

        self.last_touch = "left" if start_side == 1 else "right"
        self.entered_right = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _check_collision(self, side):
        player_pos = self.left_pos if side == "left" else self.right_pos
        player_vel = self.left_vel if side == "left" else self.right_vel

        # Calculate quickest distance between ball and player
        dist = np.linalg.norm(self.ball_pos - player_pos)

        if dist < self.collision_dist:
            normal = (self.ball_pos - player_pos) / dist
            relative_vel = self.ball_vel - player_vel
            reflect_vel = relative_vel - 2 * np.dot(relative_vel, normal) * normal

            self.ball_vel = reflect_vel + player_vel

            overlap = self.collision_dist - dist

            self.ball_pos += overlap * normal

            return True

        return False

    def _check_termination(self):
        ball_out_of_bounds = (self.ball_pos[0] <= self.ball_radius or self.ball_pos[
            0] >= 2 - self.ball_radius) or self.ball_pos[1] <= self.ball_radius
        ball_hits_net = (self.ball_pos[1] >= 0.5 and 1 - self.ball_radius <= self.ball_pos[
            0] <= 1 + self.ball_radius)
        ball_hits_ground = self.ball_pos[1] >= 1 - self.ball_radius

        winner = None

        if ball_out_of_bounds or ball_hits_net:
            winner = "right" if self.last_touch == "left" else "left"

        if ball_hits_ground:
            winner = "right" if self.ball_pos[0] < 1 else "left"

        terminated = winner is not None

        return terminated, winner

    def step(self, action, right_action=None):
        left_action = action
        if right_action is None and self.opponent is not None:
            right_action = int(self.opponent.predict(self.get_obs_mirror())[0])
        elif right_action is None:
            right_action = self.action_space.sample()

        # Update player velocities
        self.left_vel = self._action_to_direction[left_action]
        self.right_vel = self._action_to_direction[self._mirror_action[right_action]]

        # Limit player velocities if they are hugging the net
        if self.left_pos[0] == 1 and self.left_vel[0] > 0:
            self.left_vel[0] = 0
        if self.right_pos[0] == 1 and self.right_vel[0] < 0:
            self.right_vel[0] = 0

        # Apply gravity
        self.ball_vel += self.gravity

        # Check and handle collisions
        if self._check_collision("left"):
            self.last_touch = "left"
        elif self._check_collision("right"):
            self.last_touch = "right"

        # Limit ball velocity
        total_vel = np.linalg.norm(self.ball_vel)
        if total_vel > self.ball_speed:
            self.ball_vel *= self.ball_speed / total_vel

        # Update ball position
        self.ball_pos += self.ball_vel

        # Update and limit player positions
        self.left_pos += self.left_vel
        self.right_pos += self.right_vel

        self.left_pos[0] = np.clip(self.left_pos[0], self.player_radius, 1 - self.player_radius)
        self.left_pos[1] = np.clip(self.left_pos[1], self.player_radius, 1 - self.player_radius)
        self.right_pos[0] = np.clip(self.right_pos[0], 1 + self.player_radius,
                                    2 - self.player_radius)
        self.right_pos[1] = np.clip(self.right_pos[1], self.player_radius, 1 - self.player_radius)

        observation = self._get_obs()

        reward = 0

        # Reward for hitting the to the opponent's side of the court
        # if self.ball_pos[0] > 1 and self.entered_right is False:
        #    reward += 0.1
        #    self.entered_right = True
        # elif self.ball_pos[0] < 1 and self.entered_right is True:
        #    self.entered_right = False

        terminated, winner = self._check_termination()
        truncated = False

        info = {}

        if terminated:
            reward += 1 if winner == "left" else -1
            info = {"winner": winner}
        else:
            truncated = self.steps >= self.max_steps
            if truncated:
                winner = "right" if self.ball_pos[0] < 1 else "left"
                reward += 1 if winner == "left" else -1
                info = {"winner": winner}
            else:
                self.steps += 1

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size * 2, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size * 2, self.window_size))
        canvas.fill((0, 0, 0))

        pygame.draw.rect(canvas, (255, 255, 255), (0, 0, 800, 400))
        pygame.draw.rect(canvas, (0, 0, 0), (1, 1, 798, 398))
        pygame.draw.line(canvas, (255, 255, 255), (400, 400), (400, 200))

        n2p = lambda vec: (int(vec[0] * 400), int(vec[1] * 400))
        scale = lambda val: int(val * 400)

        pygame.draw.circle(canvas, (255, 255, 255), n2p(self.ball_pos), scale(self.ball_radius))
        pygame.draw.circle(canvas, (255, 0, 0), n2p(self.left_pos), scale(self.player_radius))
        pygame.draw.circle(canvas, (0, 0, 255), n2p(self.right_pos), scale(self.player_radius))

        # Draw velocity vectors out of the ball and players
        pygame.draw.line(canvas, (0, 255, 0), n2p(self.ball_pos),
                         n2p(self.ball_pos + self.ball_vel * 10), width=3)
        pygame.draw.line(canvas, (255, 0, 255), n2p(self.left_pos),
                         n2p(self.left_pos + self.left_vel * 10), width=3)
        pygame.draw.line(canvas, (255, 0, 255), n2p(self.right_pos),
                         n2p(self.right_pos + self.right_vel * 10), width=3)

        # Print time in lower left corner
        font = pygame.font.Font(None, 36)
        text = font.render(f"{self.steps}/{self.max_steps}", True, (255, 255, 255))
        canvas.blit(text, (10, 360))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
