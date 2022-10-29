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
    parser.add_argument("--save_name",default='temp_model')
    parser.add_argument("--algo",default='PPO')
    parser.add_argument("--time_steps", default=1e4,type=int)
    args = vars(parser.parse_args())

    #init log
    logging.basicConfig(level=logging.INFO, format='%(filename)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Log Initilized Successfully!")

    #init env
    logger.info("Start Create Envs...")
    env = gym.make("gym_examples/"+str(args["game_name"]),render_mode='human')
    logger.info("Envs created Successfully!")
    check_env(env, warn=True)
    logger.info("Envs satisfy the requirement")


    # train
    logger.info("Start training with "+str(args["algo"])+"...")
    total_time_steps = int(args["time_steps"])
    if args["algo"] == PPO:
        model = PPO('MlpPolicy', env, verbose=1, n_steps=16, gae_lambda=0.8, gamma=0.98, n_epochs=4, ent_coef=0.0,
                    batch_size=256).learn(total_time_steps)
    else:
        model = A2C('MlpPolicy', env, verbose=1, n_steps=16, gae_lambda=0.8, gamma=0.98, ent_coef=0.0,
                   ).learn(total_time_steps)

    logger.info("Finished Training after " + str(args["time_steps"]) + " time steps!")

    # Create save dir
    root_dir = os.getcwd()
    model_name = args["save_name"]
    save_dir = root_dir + "/models/" + model_name
    os.makedirs(root_dir, exist_ok=True)
    model.save(save_dir)
    logger.info("Model successfully saved as: " + model_name + ".zip !"+"at "+save_dir)
