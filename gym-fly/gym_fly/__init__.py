from gym.envs.registration import register

register(
    id='fly-v1',
    entry_point='gym_fly.envs:FlyEnv',
)