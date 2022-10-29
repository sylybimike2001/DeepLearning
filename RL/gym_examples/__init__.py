from gym.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)

register(
    id="gym_examples/CollisionAvoid-v0",
    entry_point="gym_examples.envs:CollisionAvoid",
)

register(
    id="gym_examples/GoLeftEnv-v0",
    entry_point="gym_examples.envs:GoLeftEnv",
)

register(
    id="gym_examples/CollisionAvoid-v1",
    entry_point="gym_examples.envs:CollisionAvoid_V1",
)

register(
    id="gym_examples/CollisionAvoid-v2",
    entry_point="gym_examples.envs:CollisionAvoid_V2",
)