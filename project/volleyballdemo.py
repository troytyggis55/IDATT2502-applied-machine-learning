from random import random
import pygame
import numpy as np

# This is a pure pygame demo to test the environment. Player left uses wasd and player right uses
# arrow keys.

# Layout: w = 800, h = 400
# Ball: x = 400, y = 100 x_vel = random(-1, 1), y_vel = random(-1, 0)
# Player 1: x = 200, y = 200
# Player 2: x = 600, y = 200

# Actions are 0: do nothing, 1: move left, 2: move right, 3: move up, 4: move down Observations
# are 0: ball_x, 1: ball_y, 2: ball_x_vel, 3: ball_y_vel, 4: left_x, 5: left_y, 6: right_x,
# 7: right_y

player_radius = 10
ball_radius = 20
terminal_velocity = 6
player_speed = 4
gravity = 0.1

time_step = 0.3

class Player:
    def __init__(self, x, y, side):
        self.x = x
        self.y = y
        self.x_vel = 0
        self.y_vel = 0
        self.side = side

    def update_position(self, action):
        diagonal_velocity = player_speed / np.sqrt(2)
        self.x_vel = 0
        self.y_vel = 0

        if action == 1:
            self.x_vel = -player_speed
        elif action == 2:
            self.x_vel = player_speed
        elif action == 3:
            self.y_vel = -player_speed
        elif action == 4:
            self.y_vel = player_speed
        elif action == 5:
            self.x_vel = -diagonal_velocity
            self.y_vel = -diagonal_velocity
        elif action == 6:
            self.x_vel = diagonal_velocity
            self.y_vel = -diagonal_velocity
        elif action == 7:
            self.x_vel = -diagonal_velocity
            self.y_vel = diagonal_velocity
        elif action == 8:
            self.x_vel = diagonal_velocity
            self.y_vel = diagonal_velocity

        self.x += self.x_vel * time_step
        self.y += self.y_vel * time_step

        # Limit player to its area
        if self.side == "left":
            if self.x < 0 + player_radius:
                self.x = 0 + player_radius
            elif self.x > 400 - player_radius:
                self.x = 400 - player_radius
        elif self.side == "right":
            if self.x < 400 + player_radius:
                self.x = 400 + player_radius
            elif self.x > 800 - player_radius:
                self.x = 800 - player_radius

        if self.y < 0 + player_radius:
            self.y = 0 + player_radius
        elif self.y > 400 - player_radius:
            self.y = 400 - player_radius


class Volleyball:
    def __init__(self):
        self.ball_x = None
        self.ball_y = None
        self.ball_x_vel = None
        self.ball_y_vel = None

        self.left_player = Player(200, 200, "left")
        self.right_player = Player(600, 200, "right")

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
        

        self.left_player.x = 200
        self.left_player.y = 200

        self.right_player.x = 600
        self.right_player.y = 200

    def step(self, actions):
        # Get actions
        left_action = actions[0]
        right_action = actions[1]

        # Move players
        self.left_player.update_position(left_action)
        self.right_player.update_position(right_action)

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
        def check_collision(player, ball_x, ball_y, ball_vel):
            # Calculate the squared distance between the ball and the player
            dist_sq = (player.x - ball_x) ** 2 + (player.y - ball_y) ** 2
            collision_dist_sq = (player_radius + ball_radius) ** 2

            if dist_sq < collision_dist_sq:
                # Calculate normal vector from agent to ball
                normal = [(ball_x - player.x) / np.sqrt(dist_sq),
                          (ball_y - player.y) / np.sqrt(dist_sq)]

                # Reflect the ball velocity considering the agent's velocity
                new_ball_vel = reflect(ball_vel, normal, [player.x_vel, player.y_vel])

                # Move ball to the edge of the collision distance to prevent duplicate hits
                overlap = (player_radius + ball_radius) - np.sqrt(dist_sq)
                ball_x += normal[0] * overlap
                ball_y += normal[1] * overlap

                self.last_touch = player.side

                return ball_x, ball_y, new_ball_vel

            return ball_x, ball_y, ball_vel

        # Update ball position and velocity based on collisions with players
        self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel] = check_collision(
            self.left_player, self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel]
        )

        self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel] = check_collision(
            self.right_player, self.ball_x, self.ball_y, [self.ball_x_vel, self.ball_y_vel]
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
        
        distance_from_net = (self.ball_x - 400) ** 2 + (self.ball_y - 200)**2
        collision_dist_sq = ball_radius ** 2
                                
        ball_hits_net = (self.ball_y >= 200 and self.ball_x >= 400 - ball_radius 
                         and self.ball_x <= 400 + ball_radius or distance_from_net <= 
                         collision_dist_sq)
        
        # Ball collision with top and walls
        if (self.ball_y <= ball_radius or self.ball_x <= ball_radius or self.ball_x >= 800 - 
                ball_radius) or ball_hits_net:
            if self.last_touch == "left":
                # Right player wins
                print("Right player wins")
                self.close()
                pass
            else:
                # Left player wins
                print("Left player wins")
                self.close()
                pass
                
    
        # Ball collision with ground
        if self.ball_y >= 400 - ball_radius:
            if self.ball_x < 400:
                # Right player wins
                print("Right player wins")
                self.close()
                pass
            else:
                # Left player wins
                print("Left player wins")
                self.close()
                pass


    def render(self):
        self.screen.fill((0, 0, 0))

        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, 800, 400))
        pygame.draw.rect(self.screen, (0, 0, 0), (1, 1, 798, 398))
        pygame.draw.line(self.screen, (255, 255, 255), (400, 400), (400, 200))

        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)),
                           ball_radius)
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (int(self.left_player.x), int(self.left_player.y)), player_radius)
        pygame.draw.circle(self.screen, (0, 0, 255),
                           (int(self.right_player.x), int(self.right_player.y)), player_radius)

        pygame.display.flip()

    def close(self):
        pygame.quit()


env = Volleyball()

env.reset()

step_count = 0

while True:
    step_count += 1
    print(int(step_count * time_step))
    
    env.render()
    #print(np.sqrt(env.ball_x_vel ** 2 + env.ball_y_vel ** 2))

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
