import gym
import model
import numpy as np
import random
env = gym.make('CartPole-v0')

"""
This example is model based RL
first, find transition model from samples

check "Model-free policy iteration" from 
http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf

If we want model free control, we have to change it into Q function(off policy) based.

"""

def collect_transition_samples(num):
    samples = []
    state = env.reset()
    for i in range(num):
        lastState = state
        action = env.action_space.sample()    
        state, reward, done, info = env.step(action)    
        if done:
            state = env.reset()
        samples.append([lastState, action, state])
    return samples

transitionProbNet = model.TransitionPropNet()
transitionProbNet.train(collect_transition_samples(100000), epochs=10)

"""
then do greedy policy improvement with Monte carlo method
"""
def collect_episod_samples(num=1000):
    samples = []
    for _ in range(num):

        state = env.reset()
        # play episod
        
        history = [state]
        for j in range(100):            
            action = env.action_space.sample() # take random action            
            state, reward, done, info = env.step(action)
            history.append(state)
            if done:                
                break
        
        length = len(history) 
        randIdx = random.randint(0, length-1)
        g = len(history)-randIdx
        s = history[randIdx]

        samples.append([s, g])

    return samples
    
mcValueNet = model.McValueNet()
mcValueNet.train(collect_episod_samples(50000), epochs=50)

# play greedy
actions = [0, 1]
for _ in range(1000):    
    state = env.reset()

    for i in range(1000):
        env.render()

        nextState = transitionProbNet.predict([state]*len(actions), actions)
        values = mcValueNet.predict(nextState)
        action = np.argmax(np.reshape(values, [-1]))

        state, reward, done, info = env.step(action)
        if done:
            print('done', i)
            break
