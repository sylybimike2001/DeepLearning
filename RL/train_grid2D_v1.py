import gym_examples
import gym
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def check_mode_saved(model):
    # sample an observation from the environment
    obs = model.env.observation_space.sample()

    # Check prediction before saving
    print("pre saved", model.predict(obs, deterministic=True))

    del model  # delete trained model to demonstrate loading

    loaded_model = PPO.load(save_dir + "/PPO_tutorial")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", loaded_model.predict(obs, deterministic=True))
    if np.array_equal(model.predict(obs, deterministic=True),loaded_model.predict(obs, deterministic=True)):
        return True
    else :
        return False

if __name__ == '__main__':
    # init
    env = gym.make("gym_examples/CollisionAvoid-v1",render_mode='human')
    # env = make_vec_env("gym_examples/CollisionAvoid-v0",n_envs=8)
    check_env(env,warn=True)
    print("Check Finished,start training...")

    # train
    model = PPO('MlpPolicy', env,verbose=1,n_steps=16,gae_lambda=0.8,gamma=0.98,n_epochs=4,ent_coef=0.0,batch_size=256).learn(10000)


print("Finish training,start saving")

# Create save dir
save_dir = "/tmp/gym/"
model_name = "/CollisionAvoid-Fixed-dis-PPO-10000"
os.makedirs(save_dir, exist_ok=True)

# The model will be saved under PPO_tutorial.zip
model.save(save_dir + model_name)
print("Model saved as:" + model_name)

