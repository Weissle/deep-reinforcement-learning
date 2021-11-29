import gym

from ppo2.agent import Agent
from config.config import config as cfg

def really_play(env,agent):
	pre_observation = env.reset()
	agent.reset()
	total_reward = 0
	done = False
	while done == False:
		action = agent.predict(pre_observation)
		obs,reward,done,info = env.step(action)
		pre_observation = obs
		env.render()
		total_reward += reward
	print(f'predict total reward: {total_reward}')

def main():
	env = gym.make(cfg.env_cfg.name)
	cfg.env_cfg.state_dim = env.observation_space.shape[0]
	cfg.env_cfg.act_dim = env.action_space.n
	player1 = Agent(cfg)

	while(True):
		really_play(env,player1)
if __name__ == '__main__':
	main()
