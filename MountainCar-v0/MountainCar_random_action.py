import gym 
import gym.spaces

env = gym.make("MountainCar-v0")

for i in range(1000):
	observation = env.reset()
	for j in range(200):
		env.render()
		action = env.action_space.sample()
		obeservation, reward, done, info = env.step(action)
		print(action)
		if done:
			break
