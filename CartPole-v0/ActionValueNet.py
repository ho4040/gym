import tensorflow as tf
import numpy as np

class ActionValueNet:
	"Q function approximatior"
	def __init__(self, learning_rete=0.001):
		self.build(learning_rete)

	def build(self, learning_rete=0.001):
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Dense(units=128, activation='relu'))
		self.model.add(tf.keras.layers.Dense(units=128, activation='relu'))
		self.model.add(tf.keras.layers.Dense(units=1, activation=None))
		self.model.compile(optimizer='adam', loss='mean_squared_error')

	def train(self, samples, labels, epochs):
		states = np.array([s['state'] for s in samples])
		actions = np.array([[s['action']] for s in samples])
		x = np.concatenate([states, actions], axis=1)
		self.model.fit(x, np.array(labels), epochs=epochs, verbose=1)
	
	def predict(self, state, actions):
		states = np.array([state.tolist()] * len(actions))
		actions = np.reshape(np.array(actions), [-1, 1])
		x = np.concatenate([states, actions], axis=1)
		return self.model.predict(x)
