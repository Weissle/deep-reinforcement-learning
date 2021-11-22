import torch
import torch.nn as nn

class StateExtractor(nn.Module):
	
	def __init__(self):
		pass

class Critic(nn.Module):

	def __init__(self,state_dim):
		super(Critic, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(state_dim,64),
			nn.LeakyReLU(),
			nn.Linear(64,64),
			nn.LeakyReLU(),
			nn.Linear(64,1)
		)

	def forward(self,x):
		return self.fc(x)

	
class Actor(nn.Module):

	def __init__(self,state_dim,act_dim):
		super(Actor, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(state_dim,64),
			nn.LeakyReLU(),
			nn.Linear(64,64),
			nn.LeakyReLU(),
			nn.Linear(64,act_dim),
			nn.Softmax(-1)
		)

	def forward(self,x):
		return self.fc(x)

		