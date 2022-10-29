import gym
import os
import gym_examples
import numpy as np
from  stable_baselines3 import PPO

save_dir = "/tmp/gym/"
model = PPO.load(save_dir + "/CollisionAvoid-Fixed-PPO-1e5")
# Check that the prediction is the same after loading (for the same observation)
# print("loaded", loaded_model.predict(obs, deterministic=True))

# Test the trained agent
env = gym.make("gym_examples/CollisionAvoid-v0",render_mode='human',mode='test')
for epoch in range(30):
    print("Epoch {}".format(epoch+1))
    obs = env.reset()
    n_steps = 20
    for step in range(n_steps):
        my_action, _ = model.predict(obs, deterministic=True)
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(my_action)
        env.render()
        if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break