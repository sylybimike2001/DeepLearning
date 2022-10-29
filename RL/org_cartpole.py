import gym,os
from stable_baselines3 import A2C,PPO
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

flag = 'tEST'
if flag == 'train':
    # env = gym.make('CartPole-v1')
    env = make_vec_env('MountainCar-v0',n_envs=16)
    model = PPO("MlpPolicy", env,normalize_advantage=True,n_steps=16,gae_lambda=0.98,gamma=0.99,
                n_epochs=4,ent_coef=0.0,verbose=1,batch_size=256)
    model.learn(int(3e5))
    """
    # Tuned
    MountainCar-v0:
  normalize: true
  n_envs: 16
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  n_steps: 16
  gae_lambda: 0.98
  gamma: 0.99
  n_epochs: 4
  ent_coef: 0.0
      """

    # Create save dir
    save_dir = "/tmp/gym/"
    os.makedirs(save_dir, exist_ok=True)

    # The model will be saved under PPO_tutorial.zip
    model.save(save_dir + "/CartPole-v1")

    obs = env.reset()

    quit(0)

else:
    # test
    import gym
    import os
    import gym_examples
    import numpy as np
    from  stable_baselines3 import PPO

    save_dir = "/tmp/gym/"
    model = PPO.load(save_dir + "/CartPole-v1")
    # Check that the prediction is the same after loading (for the same observation)
    # print("loaded", loaded_model.predict(obs, deterministic=True))

    # Test the trained agent
    env = gym.make('MountainCar-v0')
    for epoch in range(10):
        print("Epoch {}".format(epoch+1))
        obs = env.reset()
        n_steps = 100000
        done = False
        while not done:
            my_action, _ = model.predict(obs)
            obs, reward, done, info = env.step(my_action)
            print(obs)
            env.render()