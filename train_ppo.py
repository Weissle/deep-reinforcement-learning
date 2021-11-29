import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym

# import pybullet_envs

from ppo2.agent import Agent
from config.config import config as cfg_
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
def really_play(env,agent):
    obs = env.reset()
    done = False
    reward =  0
    while not done:
        action = agent.predict(obs)
        env.render()
        s,r,done,_ = env.step(action)
        reward += r
        obs = s
    print(f'really_play reward : {reward}')

def train():
    cfg = deepcopy(cfg_)
    cfg.logs.tensorboard = SummaryWriter(log_dir=f'runs/ppo_ret_norm')
    # env_name = "CartPole-v1"
    # env_name = "LunarLander-v2"
    env_name = cfg.env_cfg.name

    env = gym.make(env_name)
    cfg.env_cfg.state_dim = env.observation_space.shape[0]
    cfg.env_cfg.act_dim = env.action_space.n

    ppo_agent = Agent(cfg)

    tensorboard = cfg.logs.tensorboard

    # training loop
    total_reward = 0
    for epoch in range(1,1+cfg.train.max_episodes):

        state = env.reset()
        current_ep_reward = 0

        done = False
        while not done:

            # select action with policy
            action = ppo_agent.sample(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            current_ep_reward += reward

        total_reward += current_ep_reward
        if epoch % cfg.logs.reward_write_freq == 0:
            if tensorboard != None:
                tensorboard.add_scalar('reward',total_reward/cfg.logs.reward_write_freq,int(epoch/cfg.logs.reward_write_freq))
            else:
                print(f'average reward: {total_reward/cfg.logs.reward_write_freq}')
            total_reward = 0


        if ppo_agent.buffer_size() > cfg.train.batch_size:
            ppo_agent.learn()

        if epoch % cfg.train.render_freq == 0:
            # really_play(env,ppo_agent)
            pass

        if epoch % cfg.train.save_model_freq == 0:
            ppo_agent.save()

    env.close()

if __name__ == '__main__':

    train()
    