import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class VolleyballEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "volleyball_v0", "render_fps": 25}

    def __init__(self, render_mode=None, player_radius=0.05, ball_radius=0.05, player_speed=0.01, ball_speed=0.01, gravity=2e-4, window_height=400):
        # Ball X, Ball Y, Ball X Velocity, Ball Y Velocity, Left Player X, Left Player Y, Right Player X, Right Player Y
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, -ball_speed, -ball_speed, 0.0, 0.0, 1.0, 0.0]),
                                            high=np.array([2.0, 1.0, ball_speed, ball_speed, 1.0, 1.0, 2.0, 1.0]), dtype=np.float64)
        
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
            3: 4,
            4: 3
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
        
    def _get_obs(self):
        return np.array([self.ball_pos[0], self.ball_pos[1], self.ball_vel[0], self.ball_vel[1],
                         self.left_pos[0], self.left_pos[1], self.right_pos[0], self.right_pos[1]])
    
    def _get_obs_mirror(self):
        return np.array([2 - self.ball_pos[0], self.ball_pos[1], -self.ball_vel[0], self.ball_vel[1],
                         2 - self.right_pos[0], self.right_pos[1], 2 - self.left_pos[0], self.left_pos[1]])
    
    def _get_info(self):
        return {"touch": self.last_touch}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start_rad = np.random.uniform(-np.pi / 4, 0) # - pi/8 to 0
        start_side = -1#np.random.choice([-1, 1])
        
        self.ball_pos = np.array([1, 0.25])
        self.ball_vel = np.array([start_side * self.ball_speed * np.cos(start_rad),
                                  self.ball_speed * np.sin(start_rad)])
        
        self.left_pos = np.array([0.5, 0.5])
        self.left_vel = np.array([0, 0])
        self.right_pos = np.array([1.5, 0.5])
        self.right_vel = np.array([0, 0])
        
        self.last_touch = "left" if start_side == 1 else "right"
        
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
            self.ball_vel = self.ball_vel - 2 * np.dot(self.ball_vel, normal) * normal
            self.ball_vel += player_vel * np.dot(self.ball_vel, normal)
            overlap = self.collision_dist - dist
            
            self.ball_pos += overlap * normal

            return True

        return False
    
    def _check_termination(self):
        ball_out_of_bounds = (self.ball_pos[0] <= self.ball_radius or self.ball_pos[0] >= 2 - self.ball_radius) or self.ball_pos[1] <= self.ball_radius
        ball_hits_net = (self.ball_pos[1] >= 0.5 and 1 - self.ball_radius <= self.ball_pos[0] <= 1 + self.ball_radius)
        ball_hits_ground = self.ball_pos[1] >= 1 - self.ball_radius

        winner = None

        if ball_out_of_bounds or ball_hits_net:
            winner = "right" if self.last_touch == "left" else "left"

        if ball_hits_ground:
            winner = "right" if self.ball_pos[0] < 1 else "left"

        terminated = winner is not None

        return terminated, winner
    
    def step(self, action):
        left_action = action
        right_action = 0
        
        # Update player velocities
        self.left_vel = self._action_to_direction[left_action]
        self.right_vel = self._action_to_direction[self._mirror_action[right_action]]
        
        # Limit player velocities if they are hugging the net
        if self.left_pos[0] == 0.5 and self.left_vel[0] > 0:
            self.left_vel[0] = 0
        if self.right_pos[0] == 0.5 and self.right_vel[0] < 0:
            self.right_vel[0] = 0
        
        # Apply gravity
        self.ball_vel += self.gravity
        
        reward = 0
        
        # Check and handle collisions
        if self._check_collision("left"):
            reward += 0.5
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
        
        self.left_pos[0] = np.clip(self.left_pos[0], 0, 1)
        self.left_pos[1] = np.clip(self.left_pos[1], 0, 1)
        self.right_pos[0] = np.clip(self.right_pos[0], 1, 2)
        self.right_pos[1] = np.clip(self.right_pos[1], 0, 1)
        
        observation = self._get_obs()
        terminated, winner = self._check_termination()
        
        if terminated:
            reward += 1 if winner == "left" else -1
            info = {"winner": winner}
        else:
            info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

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

        pygame.draw.circle(canvas, (255, 255, 255), (int(self.ball_pos[0] * 400), int(self.ball_pos[1] * 400)), self.ball_radius * 400)
        pygame.draw.circle(canvas, (255, 0, 0), (int(self.left_pos[0] * 400), int(self.left_pos[1] * 400)), self.player_radius * 400)
        pygame.draw.circle(canvas, (0, 0, 255), (int(self.right_pos[0] * 400), int(self.right_pos[1] * 400)), self.player_radius * 400)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()