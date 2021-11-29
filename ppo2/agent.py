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
		self.tensorboard = cfg.logs.tensorboard
		self.learn_times = 0



	def sample(self,state):
		state = torch.FloatTensor(state).to(self.device)
		action_prob = self.algorithm.predict(state)

		dist = Categorical(action_prob)

		action = dist.sample()
		action_logprob = dist.log_prob(action).detach()

		self.buffer.value.append(self.algorithm.value(state).cpu())
		self.buffer.states.append(state.cpu())
		self.buffer.actions.append(action.cpu())
		self.buffer.logprobs.append(action_logprob.cpu())

		return action.item()

	def predict(self,state):
		state = torch.FloatTensor(state).to(self.device)
		return torch.argmax(self.algorithm.predict(state)).item()


	def learn(self):
		loss = self.algorithm.learn(self.buffer)
		if self.tensorboard != None:
			self.tensorboard.add_scalar('loss',loss,self.learn_times)
		else:
			print(f'loss : {loss}')
		self.learn_times += 1
		self.buffer.reset()
		# self.greedy = max(self.greedy-self.greedy_decrease,self.greedy_min)

	def put_reward_done_data(self,reward,done):
		self.buffer.rewards.append(reward)
		self.buffer.is_terminals.append(done)

	def save(self):
		self.algorithm.save()

	def buffer_size(self):
		return len(self.buffer.states)

	def reset(self):
		pass
	

