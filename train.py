import gym

from ddpg.agent import Agent
from config.config import config as cfg
from utils.replaybuffer import ReplayBuffer

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
	env = gym.make("CartPole-v1")
	cfg.env_cfg.state_dim = env.observation_space.shape[0]
	cfg.env_cfg.act_dim = env.action_space.n
	player1 = Agent(cfg)

	for i in range(cfg.train.max_episodes):
		pre_observation = env.reset()
		player1.reset()
		total_reward  = 0
		step=0
		done = False
		while not done:
			action = player1.sample(pre_observation)
			obs,reward,done,info = env.step(action)
			ter = 1 if done else 0
			player1.put_reward_done_data(reward,ter)
			pre_observation = obs
			total_reward += reward
			step+=1
		player1.learn()
		print(f'round {i}, step : {step}, total reward: {total_reward}')


		if i%50 == 0:
			really_play(env, player1)
			if i != 0:
				player1.save()
if __name__ == '__main__':
	main()