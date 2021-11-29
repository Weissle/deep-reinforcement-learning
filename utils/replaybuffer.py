from collections import deque
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math
import random

class ReplayBuffer:

	def __init__(self,capacity):
		self.memory = list()
		self.capacity = capacity
		self.position = 0

	def append(self,*x):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = x
		self.position = (self.position + 1) % self.capacity
	
	def sample(self,batch_size):
		return random.sample(self.memory,batch_size)

	def __len__(self):
		return len(self.memory)

	def reset(self):
		self.memory = []
		self.position = 0