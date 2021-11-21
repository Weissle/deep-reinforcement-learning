from easydict import EasyDict
import random
import numpy as np
import torch
from torch.distributions import Categorical

from .ppo import PPO

import sys
sys.path.append('../')
from utils.rolloutbuffer import RolloutBuffer

class Agent:
	
	def __init__(self, cfg:EasyDict):
		self._cfg = cfg

		self.device = cfg.device
		self.algorithm = PPO(cfg)

		self.buffer = RolloutBuffer()

	def sample(self,observation):
		observation = torch.Tensor(observation).float()
		prob = self.algorithm.predict(observation.to(self.device))
		dist = Categorical(prob)		
		action = dist.sample()
		action_logprob = dist.log_prob(action)
		
		self.buffer.states.append(observation)
		self.buffer.actions.append(action)
		self.buffer.logprobs.append(action_logprob)
		return action.item()

	def predict(self,observation):
		observation = torch.Tensor(observation).float()
		return torch.argmax(self.algorithm.predict(observation.to(self.device))).item()

	def reset(self):
		self.algorithm.reset()
		self.buffer.reset()

	def learn(self):
		self.algorithm.learn(self.buffer)
		self.algorithm.sync_target(1)

	def put_reward_done_data(self,reward,done):
		self.buffer.rewards.append(reward)
		self.buffer.is_terminals.append(done)

	def save(self):
		self.algorithm.save()
