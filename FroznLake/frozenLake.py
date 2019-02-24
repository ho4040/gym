import gym
env = gym.make('FrozenLake-v0')

for j in range(100):
	state = env.reset()
	for i in range(1000):
		env.render()
		action = env.action_space.sample()
		state, reward, done, info = env.step(action)
		print("%d %d %d" % (state, reward, done))
		if done:
			print("done in %d"%i)
			break
		elif i == 999:
			print("clear")
