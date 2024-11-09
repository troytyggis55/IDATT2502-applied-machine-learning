import functools
from copy import copy

import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test


class PongEnvironment(ParallelEnv):
    name = "2PlayerPong"
    input_size = 6
    output_size = 3
    
    def __init__(self, max_bounces=100, max_steps=1000):
        self.render_mode = None
        self.screen = None
        self.width = 800
        self.height = 400
        self.aspect_ratio = self.width / self.height
        
        self.possible_agents = [0, 1]
        self.agents = self.possible_agents[:]
        self.max_bounces = max_bounces
        self.max_steps = max_steps
        self.timestep = None
        self.timestep_reward = 1 / max_steps
        
        self.pong_x = None
        self.pong_y = None
        self.pong_vx = None
        self.pong_vy = None
        
        self.left_paddle_y = None
        self.right_paddle_y = None

        self.pong_speed = None
        self.pong_speed_increase = 1.2
        self.paddle_speed = 0.01
        self.paddle_length = 0.25 / 2
        
        self.seed = None

    def observe(self):
        return {0: np.array([self.pong_x, self.pong_y, self.pong_vx, self.pong_vy, self.left_paddle_y, self.right_paddle_y]),
                 1: np.array([1 - self.pong_x, self.pong_y, -self.pong_vx, self.pong_vy, self.right_paddle_y, self.left_paddle_y])}
    
    def reset_game(self):
        self.pong_x = 0.5
        self.pong_y = 0.5
        self.pong_speed = 0.02

        random_state = np.random.Generator(np.random.PCG64(self.seed + self.timestep))
        start_rad = random_state.uniform(-np.pi / 16, np.pi / 16)  # - pi/4 to pi/4
        start_side = random_state.choice([-1, 1])

        self.pong_vx = start_side * self.pong_speed * np.cos(start_rad)
        self.pong_vy = self.pong_speed * np.sin(start_rad)
    
    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)

        self.timestep = 0
        self.left_paddle_y = 0.5
        self.right_paddle_y = 0.5
        
        self.seed = seed
        self.reset_game()
        
        self.agents = self.possible_agents[:]
        
        observation = self.observe()
        infos = {0: {}, 1: {}}

        if options is None:
            options = {}

        if options.get("render_mode") == "human":
            self.render_mode = "human"
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        return observation, infos
    
    def step(self, actions):
        if self.timestep >= self.max_steps:
            self.agents = []
            return self.observe(), {0: 0, 1: 0}, {0: False, 1: False}, {0: True, 1: True}, {"winner": None}
        
        self.timestep += 1
        
        left_paddle_action = actions[0]
        right_paddle_action = actions[1]
        
        if left_paddle_action == 1:
            self.left_paddle_y = min(1.0, self.left_paddle_y + self.paddle_speed)
        elif left_paddle_action == 2:
            self.left_paddle_y = max(0.0, self.left_paddle_y - self.paddle_speed)
            
        if right_paddle_action == 1:
            self.right_paddle_y = min(1.0, self.right_paddle_y + self.paddle_speed)
        elif right_paddle_action == 2:
            self.right_paddle_y = max(0.0, self.right_paddle_y - self.paddle_speed)
            
        self.pong_x += self.pong_vx / self.aspect_ratio
        self.pong_y += self.pong_vy
        
        if self.pong_y < 0:
            self.pong_y = -self.pong_y
            self.pong_vy = -self.pong_vy
        elif self.pong_y > 1:
            self.pong_y = 2 - self.pong_y
            self.pong_vy = -self.pong_vy
            
        rewards = {0: 0, 1: 0}
            
        if self.pong_x < 0:
            if self.left_paddle_y - self.paddle_length < self.pong_y < self.left_paddle_y + self.paddle_length:
                self.pong_speed *= self.pong_speed_increase
                self.pong_x = -self.pong_x
                
                edge_hit = (self.pong_y - self.left_paddle_y) / self.paddle_length
                self.pong_vx = self.pong_speed * np.cos(np.pi * edge_hit / 16)
                self.pong_vy = self.pong_speed * np.sin(np.pi * edge_hit / 16)
            else:
                self.reset_game()
                return self.observe(), {0: -1, 1: 1}, {0: True, 1: True}, {0: False, 1: False}, {"winner": 1}
        elif self.pong_x > 1:
            if self.right_paddle_y - self.paddle_length < self.pong_y < self.right_paddle_y + self.paddle_length:
                self.pong_speed *= self.pong_speed_increase
                
                self.pong_x = 2 - self.pong_x
                
                edge_hit = (self.pong_y - self.right_paddle_y) / self.paddle_length
                self.pong_vx = - self.pong_speed * np.cos(np.pi * edge_hit / 16)
                self.pong_vy = self.pong_speed * np.sin(np.pi * edge_hit / 16)
            else:
                self.reset_game()
                return self.observe(), {0: 1, 1: -1}, {0: True, 1: True}, {0: False, 1: False}, {"winner": 0}
            
        if self.render_mode == "human":
            self.render()
            
        return self.observe(), rewards, {0: False, 1: False}, {0: False, 1: False}, {}
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            
        self.screen.fill((0, 0, 0))
        
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, (self.left_paddle_y - self.paddle_length) * self.height, 10, self.paddle_length * 2 * self.height))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(self.width - 10, (self.right_paddle_y - self.paddle_length) * self.height, 10, self.paddle_length * 2 * self.height))
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.pong_x * self.width), int(self.pong_y * self.height)), 5)
        
        pygame.display.flip()
        pygame.time.delay(10)
    
    def close(self):
        if self.render_mode is not None:
            self.render_mode = None
            self.screen = None
            pygame.quit()
            return
            
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(3)

if __name__ == "__main__":
    env = PongEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)
