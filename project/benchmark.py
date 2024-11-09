from stable_baselines3.common.env_checker import check_env

from self_play.envs.volleyball_environment import VolleyballEnvironment

env = VolleyballEnvironment(render_mode="human")
check_env(env)
obs, info = env.reset()

eps = 0
for i in range(1000):
    obs, rewards, terminated, truncated, info = env.step(env.action_space.sample())
        
    if terminated or truncated:
        obs, info = env.reset()
        eps += 1
        print(f"Episode {eps} done")
        
print("Done")
