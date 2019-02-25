import gym
import policy.network as network
import numpy as np
import random

env = gym.make("CartPole-v0")

# Hyper parameters
gamma = 0.6
epsilon = 0.05
batch_size = 100
# init function approximator
q = network.ActionValueNet(0.002, decay=0.001, epoch=1)
episode = 0
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

		bootstrap_q_value = next_q_values[next_action]
		td_target = reward + gamma * bootstrap_q_value
		
		# Train
		h = q.train([{"state": state, "action": action}], [td_target])
		
		state = next_state			
		action = next_action
		
		if terminated:
			loss = np.mean(h.history['loss'])
			print("#%d, step:%f, loss:%f, epsilon %f" %
					(episode, step, loss, epsilon))
			break
	
	
