import functools

from gymnasium import spaces
from pettingzoo import ParallelEnv
import numpy as np
import pygame
import random


class Player:
    def __init__(self, x, y, side, player_speed, player_radius, player_to_net_dist):
        self.x = x
        self.y = y
        self.x_vel = 0
        self.y_vel = 0
        self.side = side
        self.player_speed = player_speed
        self.player_radius = player_radius
        self.player_to_net_dist = player_to_net_dist

    def update_position(self, action, time_delta):
        self.x_vel, self.y_vel = 0, 0

        if action == 1:  # Move left
            self.x_vel = -self.player_speed
        elif action == 2:  # Move right
            self.x_vel = self.player_speed
        elif action == 3:  # Move up
            self.y_vel = -self.player_speed
        elif action == 4:  # Move down
            self.y_vel = self.player_speed

        self.x += self.x_vel * time_delta
        self.y += self.y_vel * time_delta

        # Limit player to its area
        if self.side == "left":
            self.x = max(self.player_radius, min(self.x, 400 - self.player_radius - self.player_to_net_dist))
        elif self.side == "right":
            self.x = max(400 + self.player_radius + self.player_to_net_dist, min(self.x, 800 - self.player_radius))

        self.y = max(self.player_radius, min(self.y, 400 - self.player_radius))


class VolleyballEnvironment(ParallelEnv):
    metadata = {
        "render.modes": ["human"],
        "name": "volleyball_v0",
    }

    def __init__(self, player_radius=20, ball_radius=20, player_speed=4,
                 ball_speed=5, gravity=0.1, step_delta=1, players_to_net_dist=50):
        self.ball_x = 400
        self.ball_y = 100
        self.ball_x_vel = 4
        self.ball_y_vel = -0.6
        self.ball_radius = ball_radius
        self.ball_speed = ball_speed
        self.gravity = gravity

        self.player_radius = player_radius
        self.player_speed = player_speed
        self.player_x_rom = players_to_net_dist

        self.left_player = None
        self.right_player = None

        self.last_touch = None

        self.time_steps = 0
        self.time_steps_in_right = 0
        self.step_delta = step_delta

        self.render_mode = None

        self.possible_agents = ["left", "mirrored_right"]
        self.agents = self.possible_agents[:]

    def reset(self, seed=0, options=None):
        options = options or {}

        if options.get("render_mode") == "human":
            self.render_mode = "human"
            pygame.init()
            self.screen = pygame.display.set_mode((800, 400))
            pygame.display.set_caption("Volleyball")

        self.time_steps = 0
        self.time_steps_in_right = 0

        new_step_delta = options.get("step_delta")
        if new_step_delta is not None:
            self.step_delta = new_step_delta

        self.ball_x = 400
        self.ball_y = 100

        random.seed(hash((seed, options.get("gen", 0))))
        self.last_touch = random.choice(["left", "right"])
        start_x_vel = random.uniform(3, 5)
        self.ball_x_vel = start_x_vel if self.last_touch == "left" else -start_x_vel
        start_y_vel = random.uniform(-1.3, 0)
        self.ball_y_vel = start_y_vel
        
        total_vel = np.hypot(self.ball_x_vel, self.ball_y_vel)
        if total_vel > self.ball_speed:
            scale = self.ball_speed / total_vel
            self.ball_x_vel *= scale
            self.ball_y_vel *= scale

        self.left_player = Player(200, 200, "left", self.player_speed, self.player_radius, self.player_x_rom)
        self.right_player = Player(600, 200, "right", self.player_speed, self.player_radius, self.player_x_rom)

        self.agents = self.possible_agents[:]

        observations = self._get_observations()

        infos = {"left": {}, "mirrored_right": {}}

        return observations, infos

    def step(self, actions, mirror_right_action=True):
        left_action = actions["left"]
        right_action = actions["mirrored_right"]

        if mirror_right_action:
            mirror_action_map = {1: 2, 2: 1, 3: 3, 4: 4}  # Only horizontal and vertical moves
            right_action = mirror_action_map.get(right_action, right_action)

        self.left_player.update_position(left_action, self.step_delta)
        self.right_player.update_position(right_action, self.step_delta)
        
        self.has_left_touched = False
        
        self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel] = self._check_collision(
            self.left_player, self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel]
        )

        self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel] = self._check_collision(
            self.right_player, self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel]
        )

        # Apply gravity
        self.ball_y_vel += self.gravity * self.step_delta

        # Limit total velocity
        total_vel = np.hypot(self.ball_x_vel, self.ball_y_vel)
        if total_vel > self.ball_speed:
            scale = self.ball_speed / total_vel
            self.ball_x_vel *= scale
            self.ball_y_vel *= scale

        self.ball_x += self.ball_x_vel * self.step_delta
        self.ball_y += self.ball_y_vel * self.step_delta

        self.time_steps += 1
        ball_in_right = self.ball_x > 400
        if ball_in_right:
            self.time_steps_in_right += 1

        terminated, winner = self._check_termination()
        left_won = winner == "left"

        observations = self._get_observations()

        truncated = {"left": self.time_steps >= 1000, "mirrored_right": None}

        left_reward = 0.0
        left_reward += (self.time_steps_in_right / 1000) - 0.5 if truncated["left"] else 0.0
        left_reward += (self.time_steps / 100) if terminated["left"] else 0.0
        #left_reward += ball_in_right * 0.001
        left_reward += self.has_left_touched * 0.1
        #left_reward += left_won * 2.0 - 1 if terminated["left"] else 0.0

        rewards = {"left": left_reward, "mirrored_right": 0.0}

        infos = {"left": {"winner": winner}, "mirrored_right": {}}

        if self.render_mode == "human":
            self.render()

        self.agents = [agent for agent in self.possible_agents if not terminated[agent]]
        return observations, rewards, terminated, truncated, infos

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.render_mode = None
                pygame.quit()
                return

        self.screen.fill((0, 0, 0))

        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, 800, 400))
        pygame.draw.rect(self.screen, (0, 0, 0), (1, 1, 798, 398))
        pygame.draw.line(self.screen, (255, 255, 255), (400, 400), (400, 200))

        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)),
                           self.ball_radius)
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (int(self.left_player.x), int(self.left_player.y)), self.player_radius)
        pygame.draw.circle(self.screen, (0, 0, 255),
                           (int(self.right_player.x), int(self.right_player.y)), self.player_radius)

        pygame.display.flip()
        pygame.time.wait(10)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(5)  # Only 5 actions: nothing, left, right, up, down

    def _get_observations(self):
        x_rom = 400 - self.player_x_rom
        mirror_x_rom = 400 + self.player_x_rom
        
        return {
            "left": [
                self.ball_x / 800,
                self.ball_y / 400,
                self.ball_x_vel / self.ball_speed,
                self.ball_y_vel / self.ball_speed,
                self.left_player.x / x_rom,
                self.left_player.y / 400,
                (self.right_player.x - mirror_x_rom) / x_rom,
                self.right_player.y / 400],

            "mirrored_right": [
                (800 - self.ball_x) / 800,
                self.ball_y / 400,
                -self.ball_x_vel / self.ball_speed,
                self.ball_y_vel / self.ball_speed,
                (800 - self.right_player.x) / x_rom,
                self.right_player.y / 400,
                (800 - self.left_player.x - mirror_x_rom) / x_rom,
                self.left_player.y / 400],
        }

    def _check_collision(self, player, ball_x, ball_y, ball_vel):
        dist_sq = (player.x - ball_x) ** 2 + (player.y - ball_y) ** 2
        collision_dist_sq = (self.player_radius + self.ball_radius) ** 2

        if dist_sq < collision_dist_sq:
            if player.side == "left":
                self.has_left_touched = True
            
            normal = [(ball_x - player.x) / np.sqrt(dist_sq),
                      (ball_y - player.y) / np.sqrt(dist_sq)]
            new_ball_vel = self._reflect(ball_vel, normal, [player.x_vel, player.y_vel])

            overlap = (self.player_radius + self.ball_radius) - np.sqrt(dist_sq)
            ball_x += normal[0] * overlap
            ball_y += normal[1] * overlap

            self.last_touch = player.side

            return ball_x, ball_y, new_ball_vel

        return ball_x, ball_y, ball_vel

    def _reflect(self, ball_vel, normal, agent_vel):
        normal = np.array(normal) / np.linalg.norm(normal)
        rel_vel = np.array(ball_vel) - np.array(agent_vel)
        reflected_vel = rel_vel - 2 * np.dot(rel_vel, normal) * normal
        return reflected_vel + agent_vel

    def _check_termination(self):
        ball_out_of_bounds = (self.ball_x <= self.ball_radius or self.ball_x >= 800 -
                              self.ball_radius) or self.ball_y <= self.ball_radius
        ball_hits_net = (self.ball_y >= 200 and 400 - self.ball_radius <= self.ball_x <= 400 + self.ball_radius)
        ball_hits_ground = self.ball_y >= 400 - self.ball_radius

        winner = None

        if ball_out_of_bounds or ball_hits_net:
            winner = "right" if self.last_touch == "left" else "left"

        if ball_hits_ground:
            winner = "right" if self.ball_x < 400 else "left"

        terminated = {"left": winner is not None, "mirrored_right": None}

        return terminated, winner

    def action_mask(self, agent):
        mask = [True] * 5  # Five possible actions: nothing, left, right, up, down
        player = self.left_player if agent == "left" else self.right_player
    
        if player.side == "left":
            if player.x <= player.player_radius:
                mask[1] = False  # Mask left action
            if player.x >= 400 - player.player_radius - player.player_to_net_dist:
                mask[2] = False  # Mask right action
        else:
            if player.x <= 400 + player.player_radius + player.player_to_net_dist:
                mask[2] = False  # Mask left action
            if player.x >= 800 - player.player_radius:
                mask[1] = False  # Mask right action
    
        if player.y <= player.player_radius:
            mask[3] = False  # Mask up action
        if player.y >= 400 - player.player_radius:
            mask[4] = False  # Mask down action
    
        return mask
