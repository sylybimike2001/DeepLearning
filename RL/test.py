import argparse
import gym
import gym_examples
import logging
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name",default='CollisionAvoid-v0')
    parser.add_argument("--model_name",default='temp_model')
    parser.add_argument("--algo",default='PPO')
    parser.add_argument("--epochs", default=10,type=int)
    args = vars(parser.parse_args())

    # init log
    logging.basicConfig(level=logging.INFO, format='---------%(filename)s %(levelname)s: %(message)s---------')
    logger = logging.getLogger(__name__)
    logger.info("Log Initilized Successfully!")

    # init env
    logger.info("Start Create Envs...")
    env = gym.make("gym_examples/"+str(args["game_name"]),render_mode='human',mode='test')
    logger.info("Envs created Successfully!")
    check_env(env, warn=True)
    logger.info("Envs satisfy the requirement!")

    # Load model
    logger.info("Loading model...")
    save_dir = "/home/ayb/rl-example/DeepLearning/RL/gym_examples/models/"
    if args["algo"] == 'PPO':
        model = PPO.load(save_dir + args["model_name"])
    else:
        model = PPO.load(save_dir + args["model_name"])
    logger.info("Successfully load model: "+args["model_name"]+"!")
    logger.info("Start testing model...")

    # Test
    for epoch in range(int(args["epochs"])):
        print("Epoch {}".format(epoch + 1))
        obs = env.reset()
        n_steps = 20
        for step in range(n_steps):
            my_action, _ = model.predict(obs, deterministic=True)
            print("Step {}".format(step + 1))
            obs, reward, done, info = env.step(my_action)
            env.render()
            if done:
                print("Goal reached!", "reward=", reward)
                break
    logger.info("Finished!")
