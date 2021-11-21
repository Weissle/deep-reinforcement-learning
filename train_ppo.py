import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym

# import pybullet_envs

from ppo2.ppo import PPO
from config.config import config as cfg

def train():

    print("============================================================================================")


    ####### initialize environment hyperparameters ######

    env_name = "CartPole-v1"

    env = gym.make(env_name)

    cfg.env_cfg.state_dim = env.observation_space.shape[0]
    cfg.env_cfg.act_dim = env.action_space.n
    ppo_agent = PPO(cfg)


    # printing and logging variables

    # training loop
    total_reward = 0
    last_train_epoch = 0
    for epoch in range(cfg.train.max_episodes):

        state = env.reset()
        current_ep_reward = 0


        for step in range(cfg.train.max_episodes_length):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            current_ep_reward += reward

            if done:
                break
        
        total_reward += current_ep_reward
        
        if ppo_agent.buffer_size() > cfg.train.batch_size:
            ppo_agent.update()
            print(f'average_reward : {total_reward/(epoch-last_train_epoch)}')
            total_reward,last_train_epoch = 0,epoch

    env.close()

if __name__ == '__main__':

    train()
    