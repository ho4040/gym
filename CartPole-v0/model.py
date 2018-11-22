import tensorflow as tf
import numpy as np

class TransitionPropNet:
	def __init__(self):
		self.build()

	def build(self, learning_rate=0.001):		
		states = tf.keras.Input(shape=(4,), name='state_input')
		actions = tf.keras.Input(shape=(1,), name='action_input')
		
		x = tf.keras.layers.concatenate([states, actions], axis=1)
		x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
		x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
		# x = tf.keras.layers.Dropout(0.2)(x)
		predictions = tf.keras.layers.Dense(4, activation=None)(x)

		self.model = tf.keras.Model(inputs=[states, actions], outputs=predictions)
		self.model.compile(optimizer='adam', loss='mean_squared_error')

	def train(self, samples, epochs=1000):
		states = np.array([s[0] for s in samples])
		actions = np.array([s[1] for s in samples])
		nextStates = np.array([s[2] for s in samples])
		self.model.fit({"state_input":states, "action_input":actions}, nextStates, epochs=epochs)
	
	def predict(self, states, actions):
		return self.model.predict({"state_input":np.array(states), "action_input":np.array(actions)})

		
class McValueNet:
	def __init__(self):
		self.build()

	def build(self, learning_rate=0.001):
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Dense(units=64, activation='relu'))
		self.model.add(tf.keras.layers.Dense(units=1, activation=None))
		self.model.compile(optimizer='adam', loss='mean_squared_error' )

	def train(self, samples, epochs=1000):
		startStates = np.array([s[0] for s in samples])
		rewards = np.array([s[1] for s in samples])
		self.model.fit(startStates, rewards, epochs=epochs)
	
	def predict(self, states):
		return self.model.predict(np.array(states))