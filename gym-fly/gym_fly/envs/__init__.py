try:
    import Box2D
    from gym_fly.envs.fly_env import FlyEnv
except ImportError:
    Box2D = None