import gym
import gym.spaces
import model
import numpy as np
import random
env = gym.make('CartPole-v0')

actionValueNet = model.ActionValueNet(0.005)
actions = [0, 1]
samples = []
score = []
lastScore = 0
epsilon = 0.1

for episod in range(2000):  
	state = env.reset()
	history = []
	for i in range(200):
		if lastScore > 195:
			env.render()
		lastState = state        
		
		actionValues = actionValueNet.predict(lastState, actions)
		action = np.argmax(actionValues) if random.random() > epsilon else env.action_space.sample() # e-greedy
		state, reward, done, info = env.step(action)
		history.append([lastState, action])

		if done:
			score.append(i)
			for k in range(3):
				length = len(history) 
				randIdx = random.randint(0, length-1)
				g = length-randIdx
				s = history[randIdx]
				samples.append([{"state":s[0], "action":s[1]}, g]) # collect sample
			break
	
	if len(samples) >= 100:
		lastScore = np.mean(score)		
		print("episode:%d score:%0.2f epsilon:%0.2f"%(episod, lastScore, epsilon))

		if lastScore < 195:
			x = [s[0] for s in samples]
			y = [s[1] for s in samples]
			actionValueNet.train(x, y, 100)
		else:
			epsilon = epsilon * 0.5

		samples = []
		score = []

