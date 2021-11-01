try:
    import Box2D
    from gym_fly.envs.fly_env import FlyEnv
    from gym_fly.envs.fly_env import FlyEnvContinuous
    from gym_fly.envs.fly_delayed_env import FlyDelayedEnv
    from gym_fly.envs.fly_delayed_env import FlyDelayedEnvContinuous
    from gym_fly.envs.fly_lin_env import FlyLinEnv
    from gym_fly.envs.fly_lin_env import FlyLinEnvContinuous
    from gym_fly.envs.fly_lin_delayed_env import FlyLinDelayedEnv
#     from gym_fly.envs.fly_lin_delayed_env import FlyLinDelayedEnvContinuous
except ImportError:
    Box2D = None