from gymnasium.envs.registration import register

register(
    id="self_play/Volleyball",
    entry_point="self_play.envs.volleyball_environment:VolleyballEnvironment",
    max_episode_steps=500,
    reward_threshold=1.0,
    autoreset=True
)
