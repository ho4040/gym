import gym
import agent.network as network
import numpy as np
import random

env = gym.make("CartPole-v0")

# Hyper parameters
gamma = 0.9
epsilon = 0.05
batch_size = 5000
# init function approximator
q = network.ActionValueNet(0.001, decay=0.0001, epoch=100)
episode = 0
samples = []
for j in range(20000):
	
	epsilon = epsilon * 0.9999  # decay epsilon
	episode += 1
	
	# init episode
	step = 0	
	state = env.reset()	
	action = env.action_space.sample()
	if epsilon < random.random():
		action = np.argmax(q.predict(state, [0, 1]))

	# run episode
	while True:
		step += 1
		next_state, reward, terminated, info = env.step(action)

		# make TD target
		next_q_values = q.predict(next_state, [0, 1])

		if epsilon < random.random():
			next_action = np.argmax(next_q_values)
		else:
			next_action = env.action_space.sample()

		# bootstrap_q_value = next_q_values[next_action]
		bootstrap_q_value = np.max(next_q_values)
		td_target = reward + gamma * bootstrap_q_value
		
		# collect trainning data Train
		samples.append([state, action, td_target])
		
		
		state = next_state			
		action = next_action
		
		if terminated:
			print("#%d, step:%f, epsilon %f, samples:%d" % (episode, step, epsilon, len(samples)))
			break
	
	if batch_size < len(samples):
		h = q.train([{'state': x[0], 'action':x[1]} for x in samples], 
					[x[2] for x in samples])
		samples = []
	
	
