import gym
from ActionValueNet import ActionValueNet
import numpy as np
import random

env = gym.make("CartPole-v0")

# gamma = 0.1
# epsilon = 0.05
# episode = 0
# batch_size = 1
# epoch = 10
#learning_rate = 0.01

gamma = 0.2
epsilon = 0.05
episode = 0
batch_size = 4
epoch = 10
learning_rate = 0.02

q = ActionValueNet(learning_rate)
qualified = False
Tlist = []


for j in range(100000): 
	Tlist = []
	samples = []
	for i in range(batch_size):
		episode += 1 
		trajectory = []
		state = env.reset()
		while True:
			if qualified:
				env.render()
			action = env.action_space.sample()
			if random.random() > epsilon:
				action_probs = q.predict(state, [0, 1])
				action = np.argmax(action_probs)
				
			next_state, reward, terminated, info = env.step(action)
			trajectory.append([state, action, reward])
			state = next_state
			if terminated :
				break
		
		T = len(trajectory)
		Tlist.append(T)

		for t in range(T):
			transition = trajectory[t]
			s = transition[0]
			a = transition[1] 
			gt = sum([gamma**i * trajectory[t][2] for i in range(t, T)])
			samples.append([s, a, gt])

	avgT = np.mean(Tlist) if len(Tlist) > 0 else 0

	if qualified == False:
		qualified = True if avgT > 199 else False
		states_action = np.array([ {"state":x[0], "action":x[1]} for x in samples])
		gts = np.array([ x[2] for x in samples])
		h = q.train(states_action, gts, epoch)
		loss = np.mean(h.history['loss'])
	else:
		epsilon = 0
		print("Qualified agent!")

	epsilon = epsilon * 0.9999
	print("#%d, avgT:%d, epsilon %f" % (episode, avgT, epsilon))
	
	
