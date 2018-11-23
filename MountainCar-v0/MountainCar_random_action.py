import gym 
import gym.spaces

env = gym.make("MountainCar-v0")

for i in range(1000):
    observation = env.reset()
    for j in range(200):
        env.render()
        obeservation, reward, done, info = env.step(env.action_space.sample())
