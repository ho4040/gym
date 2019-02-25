import gym
import agent.network as network
import numpy as np
import random

env = gym.make("CartPole-v0")

# Hyper parameters
gamma = 0.5
epsilon = 0.3
batch_size = 100
# init function approximator
q = network.ActionValueNet(0.0003, decay=0.0, epoch=1)
total_episode = 0
episode = 0
samples = []
avgT = 0


for j in range(20000):
	
	epsilon = epsilon * 0.9999  # decay epsilon
	total_episode += 1
	episode += 1

	# init episode
	step = 0	
	terminated = False
	state = env.reset()	
	action = env.action_space.sample()
	if epsilon < random.random():
		action = np.argmax(q.predict(state, [0, 1]))

	# run episode
	while terminated == False:
		
		step += 1

		next_state, reward, terminated, info = env.step(action)		

		# make TD target
		next_q_values = q.predict(next_state, [0, 1])

		if epsilon < random.random():
			next_action = np.argmax(next_q_values)
		else:
			next_action = env.action_space.sample()

		bootstrap_q_value = np.max(next_q_values) # max for bellman optimality equation
		td_target = reward + gamma * bootstrap_q_value
		
		# collect train samples
		samples.append([state, action, td_target])
		
		state = next_state			
		action = next_action
	
	avgT = avgT + (1/episode)*(step - avgT) #increamental mean
	
	if len(samples) > batch_size:	
		x = [{"state":x[0], "action":x[1]} for x in samples]
		y = [x[2] for x in samples]		
		h = q.train(x, y)
		episode = 0
		samples = [] # clear samples
		loss = np.mean(h.history["loss"])
		print("#%05d, avgT:%.1f, loss:%.8f, epsilon %.4f" % (total_episode, avgT, loss, epsilon))

	
	
