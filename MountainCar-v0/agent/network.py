import tensorflow as tf
import numpy as np


class ActionValueNet:
	"Q function approximatior"
	def __init__(self, learning_rete=0.01, decay=0.0001, epoch=1):
		self.decaied_learning_rate = learning_rete
		self.decay = decay
		self.epochs = epoch
		self.build(learning_rete)

	def build(self, learning_rete=0.01):
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Dense(units=64, activation='relu'))
		self.model.add(tf.keras.layers.Dense(units=64, activation='relu'))
		self.model.add(tf.keras.layers.Dense(units=1, activation=None))
		self.opt = tf.keras.optimizers.Adam(lr=learning_rete, decay=self.decay)
		self.model.compile(optimizer=self.opt, loss='mean_squared_error')

	def train(self, samples, labels):
		states = np.array([s['state'] for s in samples])
		actions = np.array([[s['action']] for s in samples])
		x = np.concatenate([states, actions], axis=1)
		
		return self.model.fit(x, np.array(labels), epochs=self.epochs, verbose=0)
	
	def predict(self, state, actions):
		size = np.size(state)
		states = np.reshape(np.array([state.tolist()] * len(actions)), [-1, size])
		# states = np.array([state.tolist()] * len(actions))
		actions = np.reshape(np.array(actions), [-1, 1])
		x = np.concatenate([states, actions], axis=1)
		return self.model.predict(x)
