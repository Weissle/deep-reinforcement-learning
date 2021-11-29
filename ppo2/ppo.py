import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from utils.rolloutbuffer import RolloutBuffer
from .model import Actor,Critic


class PPO:
	def __init__(self, cfg):

		self.cfg = cfg
		self.gamma = cfg.gamma
		self.eps_clip = cfg.ppo.eps_clip
		self.K_epochs = cfg.ppo.K_epochs
		self.device = cfg.device
		self.use_gae = cfg.ppo.use_gae		
		self.gae_lambda = cfg.ppo.gae_lambda

		self.actor = Actor(cfg.env_cfg.state_dim, cfg.env_cfg.act_dim).to(self.device)
		self.critic = Critic(cfg.env_cfg.state_dim).to(self.device)
		self.load()


		self.optimizer = torch.optim.Adam([
						{'params': self.actor.parameters(), 'lr': cfg.actor.lr},
						{'params': self.critic.parameters(), 'lr': cfg.critic.lr}
					])

		self.MseLoss = nn.MSELoss()
		self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=cfg.warmup.milestones,gamma=cfg.warmup.gamma)


	def predict(self, state):
		with torch.no_grad():
			action = self.actor(state).detach()
			return action

	def value(self,state):
		with torch.no_grad():
			return self.critic(state).detach()

	def evaluate(self, state, action):
		action_probs = self.actor(state)
		dist = Categorical(action_probs)
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_values = self.critic(state)

		return action_logprobs,state_values,dist_entropy

	def to_tensor_normalized(self,vec:list):
		vec.reverse()
		ret = torch.tensor(vec, dtype=torch.float32).to(self.device)
		ret = (ret-ret.mean())/(ret.std()+1e-7)
		return ret

	def compute_returns(self,buffer):
		returns = []
		advantages = []

		if self.use_gae:
			gae,next_value = 0,0
			for reward,value,is_terminal in zip(reversed(buffer.rewards),reversed(buffer.value),reversed(buffer.is_terminals)):
				delta = reward + self.gamma * next_value * (1-is_terminal) - value 
				gae = delta + self.gamma * self.gae_lambda * (1-is_terminal) * gae
				advantages.append(gae)		
				returns.append(gae+value)
				next_value= value
				
		else:
			# Monte Carlo estimate of returns
			discounted_reward = 0
			for reward,value,is_terminal in zip(reversed(buffer.rewards),reversed(buffer.value),reversed(buffer.is_terminals)):
				if is_terminal:
					discounted_reward = 0
				discounted_reward = reward + (self.gamma * discounted_reward)
				returns.append(discounted_reward)
				advantages.append(discounted_reward-value)

		returns.reverse()
		advantages.reverse()
		buffer.returns = returns
		buffer.advantages = advantages

	def learn(self,buffer):

		self.compute_returns(buffer)

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(self.device)
		old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(self.device)
		old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(self.device)
		old_returns = torch.FloatTensor(buffer.returns).detach().to(self.device)
		old_advantages = torch.FloatTensor(buffer.advantages).detach().to(self.device)
		old_value = torch.FloatTensor(buffer.value).detach().to(self.device)

		old_returns = (old_returns - old_returns.mean()) / (old_returns.std() + 1e-5)
		# old_advantages = (old_advantages - old_advantages.mean()) / (old_advantages.std() + 1e-5)

		
		# Optimize policy for K epochs
		for _ in range(self.K_epochs):

			# Evaluating old actions and values
			# logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
			logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

			# match state_values tensor dimensions with rewards tensor
			state_values = torch.squeeze(state_values)
			
			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())
			

			# Finding Surrogate Loss
			surr1 = ratios * old_advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * old_advantages

			# final loss of clipped objective PPO
			loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, old_returns) - 0.01*dist_entropy

			# take gradient step
			self.optimizer.zero_grad()
			loss = loss.mean()
			loss.backward()
			self.optimizer.step()
			
		# Copy new weights into old policy
		self.lr_scheduler.step()
		return loss.item()

	def save(self):
		torch.save(self.actor.state_dict(),self.cfg.actor.save_path)
		torch.save(self.critic.state_dict(),self.cfg.critic.save_path)
		

	def load(self):
		if self.cfg.critic.load_from != None:
			self.critic.load_state_dict(torch.load(self.cfg.critic.load_from))
		if self.cfg.actor.load_from != None:
			self.actor.load_state_dict(torch.load(self.cfg.actor.load_from))

