import gym
import policy.network as network
import numpy as np
import random

env = gym.make("MountainCar-v0")
all_actions = [0,1,2]

# Hyper parameters
gamma = 0.999
epsilon = 0.99
batch_size = 1

# init function approximator
q = network.ActionValueNet(0.001, 0.0001, 10)
episode = 0
avgT = 0
for j in range(20000):

	samples = []
	avgT = 0

	for i in range(batch_size):
		episode += 1
		trajectory = []
		state = env.reset()

		# run episode
		for t in range(1000):
			action = env.action_space.sample()
			if random.random() > epsilon:
				action_probs = q.predict(state, all_actions)
				action = np.argmax(action_probs)

			next_state, reward, terminated, info = env.step(action)
			trajectory.append([state, action, reward])
			state = next_state
			if terminated:
				break

		# after one episode
		T = len(trajectory)

		for t in range(T):  # Collect trainning datas
			s = trajectory[t][0]
			a = trajectory[t][1]
			gt = sum([(gamma**i) * trajectory[t+i][2] for i in range(0, T-t)])
			samples.append([s, a, gt])

		avgT += (1-i/batch_size)*(T-avgT)

	# Train from experiences
	states_action = np.array([{"state": x[0], "action":x[1]} for x in samples])
	gts = np.array([x[2] for x in samples])
	h = q.train(states_action, gts)
	loss = np.mean(h.history['loss'])	
	epsilon = epsilon * 0.9999  # decay epsilon
	print("#%d, loss:%f, avgT:%d, epsilon %f, avgGt:%f" %
            (episode, loss, avgT, epsilon, np.mean(gts)))
