from random import random
import pygame
import numpy as np

# This is a pure pygame demo to test the environment. Player left uses wasd and player right uses arrow keys.

# Layout: w = 800, h = 400
# Ball: x = 400, y = 100 x_vel = random(-1, 1), y_vel = random(-1, 0)
# Player 1: x = 200, y = 200
# Player 2: x = 600, y = 200

# Actions are 0: do nothing, 1: move left, 2: move right, 3: move up, 4: move down
# Observations are 0: ball_x, 1: ball_y, 2: ball_x_vel, 3: ball_y_vel, 4: left_x, 5: left_y, 6: right_x, 7: right_y

player_radius = 10
ball_radius = 20
terminal_velocity = 5
player_speed = 3
gravity = 0.1

time_step = 0.3

class volleyball():
    def __init__(self):
        self.ball_x = None
        self.ball_y = None
        self.ball_x_vel = None
        self.ball_y_vel = None
        
        #Player 1
        self.left_x = None
        self.left_y = None
        self.left_x_vel = None
        self.left_y_vel = None
        
        #Player 2
        self.right_x = None
        self.right_y = None
        self.right_x_vel = None
        self.right_y_vel = None
        
        self.last_touch = None
        
        pygame.init()
        self.screen = pygame.display.set_mode((800, 400))

    def reset(self, seed=None, options=None):
        self.ball_x = 400
        self.ball_y = 100
        
        if random() < 0.5:
            self.ball_x_vel = 4
            self.last_touch = "left"
        else:
            self.ball_x_vel = -4
            self.last_touch = "right"

        self.ball_y_vel = 1
        
        self.left_x = 200
        self.left_y = 200
        
        self.right_x = 600
        self.right_y = 200

    def step(self, actions):
        # Get actions
        left_actions = actions[0]
        right_actions = actions[1]
        
        # Move left player
        diagonal_velocity = player_speed / np.sqrt(2)

        self.left_x_vel = 0
        self.left_y_vel = 0
        
        if left_actions == 1:
            self.left_x_vel = -player_speed
        elif left_actions == 2:
            self.left_x_vel = player_speed
        elif left_actions == 3:
            self.left_y_vel = -player_speed
        elif left_actions == 4:
            self.left_y_vel = player_speed
        elif left_actions == 5:
            self.left_x_vel = -diagonal_velocity
            self.left_y_vel = -diagonal_velocity
        elif left_actions == 6:
            self.left_x_vel = diagonal_velocity
            self.left_y_vel = -diagonal_velocity
        elif left_actions == 7:
            self.left_x_vel = -diagonal_velocity
            self.left_y_vel = diagonal_velocity
        elif left_actions == 8:
            self.left_x_vel = diagonal_velocity
            self.left_y_vel = diagonal_velocity
        
        self.left_x += self.left_x_vel * time_step
        self.left_y += self.left_y_vel * time_step
            
        # Limit left player to its area
        if self.left_x < 0 + player_radius:
            self.left_x = 0 + player_radius
        elif self.left_x > 400 - player_radius:
            self.left_x = 400 - player_radius
            
        if self.left_y < 0 + player_radius:
            self.left_y = 0 + player_radius
        elif self.left_y > 400 - player_radius:
            self.left_y = 400 - player_radius
            
        # Move right player
        self.right_x_vel = 0
        self.right_y_vel = 0
        
        if right_actions == 1:
            self.right_x_vel = -player_speed
        elif right_actions == 2:
            self.right_x_vel = player_speed
        elif right_actions == 3:
            self.right_y_vel = -player_speed
        elif right_actions == 4:
            self.right_y_vel = player_speed
        elif right_actions == 5:
            self.right_x_vel = -diagonal_velocity
            self.right_y_vel = -diagonal_velocity
        elif right_actions == 6:
            self.right_x_vel = diagonal_velocity
            self.right_y_vel = -diagonal_velocity
        elif right_actions == 7:
            self.right_x_vel = -diagonal_velocity
            self.right_y_vel = diagonal_velocity
        elif right_actions == 8:
            self.right_x_vel = diagonal_velocity
            self.right_y_vel = diagonal_velocity
            
        self.right_x += self.right_x_vel * time_step
        self.right_y += self.right_y_vel * time_step
        
        # Limit right player to its area
        if self.right_x < 400 + player_radius:
            self.right_x = 400 + player_radius
        elif self.right_x > 800 - player_radius:
            self.right_x = 800 - player_radius
            
        if self.right_y < 0 + player_radius:
            self.right_y = 0 + player_radius
        elif self.right_y > 400 - player_radius:
            self.right_y = 400 - player_radius

        def reflect(ball_vel, normal, agent_vel):
            # Normalize the normal vector
            normal = np.array(normal) / np.linalg.norm(normal)
        
            # Calculate relative velocity of the ball with respect to the agent
            rel_vel = np.array(ball_vel) - np.array(agent_vel)
        
            # Reflect the relative velocity along the normal
            reflected_vel = rel_vel - 2 * np.dot(rel_vel, normal) * normal
        
            # Final velocity of the ball will be reflected velocity + agent's velocity
            return reflected_vel + agent_vel
        
        # Function to handle collision, considering both ball radius and agent's velocity
        def check_collision(player_x, player_y, player_vel, ball_x, ball_y, ball_vel):
            # Calculate the squared distance between the ball and the player
            dist_sq = (player_x - ball_x) ** 2 + (player_y - ball_y) ** 2
            collision_dist_sq = (player_radius + ball_radius) ** 2
        
            if dist_sq < collision_dist_sq:
                print("Collision")
                # Calculate normal vector from agent to ball
                normal = [(ball_x - player_x) / np.sqrt(dist_sq), (ball_y - player_y) / np.sqrt(dist_sq)]
        
                # Reflect the ball velocity considering the agent's velocity
                new_ball_vel = reflect(ball_vel, normal, player_vel)
        
                # Move ball to the edge of the collision distance to prevent duplicate hits
                overlap = (player_radius + ball_radius) - np.sqrt(dist_sq)
                ball_x += normal[0] * overlap
                ball_y += normal[1] * overlap
        
                return ball_x, ball_y, new_ball_vel
        
            return ball_x, ball_y, ball_vel
        
        # Update ball position and velocity based on collisions with player 1
        self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel] = check_collision(
            self.left_x, self.left_y, [self.left_x_vel, self.left_y_vel],
            self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel]
        )
        
        # Update ball position and velocity based on collisions with player 2
        self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel] = check_collision(
            self.right_x, self.right_y, [self.right_x_vel, self.right_y_vel],
            self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel]
        )


        
        # Apply gravity
        self.ball_y_vel += gravity * time_step
        
        # Limit total velocity
        total_vel = np.sqrt(self.ball_x_vel ** 2 + self.ball_y_vel ** 2)
        if total_vel > terminal_velocity:
            self.ball_x_vel *= terminal_velocity / total_vel
            self.ball_y_vel *= terminal_velocity / total_vel
            
        
        self.ball_x += self.ball_x_vel * time_step
        self.ball_y += self.ball_y_vel * time_step
        
    def render(self):
        self.screen.fill((0, 0, 0))

        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, 800, 400))
        pygame.draw.rect(self.screen, (0, 0, 0), (1, 1, 798, 398))
        pygame.draw.line(self.screen, (255, 255, 255), (400, 400), (400, 200))
        
        
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), ball_radius)
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.left_x), int(self.left_y)), player_radius)
        pygame.draw.circle(self.screen, (0, 0, 255), (int(self.right_x), int(self.right_y)), player_radius)
        
        pygame.display.flip()
        
    def close(self):
        pygame.quit()
        
env = volleyball()

env.reset()

while True:
    env.render()

    left_action = 0
    right_action = 0
    
    # Check the state of all keys
    keys = pygame.key.get_pressed()
    
    # Determine movement for the left player based on key states
    if keys[pygame.K_w] and keys[pygame.K_a]:  # Up-Left
        left_action = 5
    elif keys[pygame.K_w] and keys[pygame.K_d]:  # Up-Right
        left_action = 6
    elif keys[pygame.K_s] and keys[pygame.K_a]:  # Down-Left
        left_action = 7
    elif keys[pygame.K_s] and keys[pygame.K_d]:  # Down-Right
        left_action = 8
    elif keys[pygame.K_w]:  # Up
        left_action = 3
    elif keys[pygame.K_s]:  # Down
        left_action = 4
    elif keys[pygame.K_a]:  # Left
        left_action = 1
    elif keys[pygame.K_d]:  # Right
        left_action = 2
    
    # Determine movement for the right player based on key states
    if keys[pygame.K_UP] and keys[pygame.K_LEFT]:  # Up-Left
        right_action = 5
    elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:  # Up-Right
        right_action = 6
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:  # Down-Left
        right_action = 7
    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:  # Down-Right
        right_action = 8
    elif keys[pygame.K_UP]:  # Up
        right_action = 3
    elif keys[pygame.K_DOWN]:  # Down
        right_action = 4
    elif keys[pygame.K_LEFT]:  # Left
        right_action = 1
    elif keys[pygame.K_RIGHT]:  # Right
        right_action = 2

    env.step([left_action, right_action])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            break

    pygame.time.wait(10)