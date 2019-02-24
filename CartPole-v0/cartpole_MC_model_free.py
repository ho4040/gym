import gym
from ActionValueNet import ActionValueNet
import numpy as np
import random

env = gym.make("CartPole-v0")
q = ActionValueNet()
epsilon = 0.05
episode = 0
batch_size = 100

for j in range(2000): 
    samples = []
    print("episode %d"%episode)
        
    for i in range(batch_size):
        
        episode += 1 
        
        trajectory = []
        state = env.reset()
        
        while True:
            if episode > 1000:
                env.render()
            action = env.action_space.sample()
            
            if random.random() > epsilon:
                action_probs = q.predict(state, [0, 1])
                action = np.argmax(action_probs)
                
            state, reward, terminated, info = env.step(action)
            trajectory.append([state, action, reward])
            if terminated :
                break
        
        for t in range(len(trajectory)):
            transition = trajectory[t]
            state = transition[0]
            action = transition[1] 
            gt = sum([tr[2] for tr in trajectory[t:]])
            samples.append([state, action, gt])

    states_action = np.array([ {"state":x[0], "action":x[1]} for x in samples])
    gts = np.array([ x[2] for x in samples])
    q.train(states_action, gts, 10)