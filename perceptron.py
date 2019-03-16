import numpy as np
import os
import random
from activations import sigmoid, sigmoid_p

# percentage and truncate functions used in displaying training completion
def truncate(x, level):
	return int(x*level)/level

def percent(x, total, level):
	return truncate((x/total)*100, level)

class perceptron:
	# initialize the Network with hyperparameters
	def __init__(self, inputs, layers, hidden, output, epochs):
		self.inputNodes = int(inputs)
		self.hiddenlayers = int(layers)
		self.hiddenNodes = int(hidden)
		self.outputNodes = int(output)
		self.epochs = epochs

		self.learningRate = .1

		self.weights = []

		for i in range(self.hiddenlayers+1):
			if i == 0:
				self.weights.append(2*np.random.rand(self.hiddenNodes, self.inputNodes)-1)
			elif i == self.hiddenlayers:
				self.weights.append(2*np.random.rand(self.outputNodes, self.hiddenNodes)-1)
			else:
				self.weights.append(2*np.random.rand(self.hiddenNodes, self.hiddenNodes)-1)

		self.biases = []

		for i in range(self.hiddenlayers+1):
			if i == self.hiddenlayers:
				self.biases.append(2*np.random.rand(self.outputNodes,1)-1)
			else:
				self.biases.append(2*np.random.rand(self.hiddenNodes, 1)-1)

	def setEpochs(self, x):
		self.epochs = x

	def getEpochs(self):
		return self.epochs

	def setLearningRate(self, x):
		self.learningRate = x

	def getLearningRate(self):
		return self.learningRate

	def fit(self, training_data):
		datasize = len(training_data)
		for i in range(self.epochs):
			index = np.random.randint(datasize)
			data = training_data[index]
			data_len = len(data)-1
			info = data[0]
			goal = data[1]
			self.train(info, goal)
			print("epoch: %d, error: %f, %g%% complete" % (i, self.mse(info, goal), percent(i, self.epochs-1, 100)))

	def train(self, inputArray, goalArray):
		inputs = np.array(inputArray)
		inputs = inputs.reshape(2, 1)
		targets = np.array(goalArray)
		targets.reshape(self.outputNodes, 1)
		layers = []
		layers.append(inputs)

		# same as the process function with a little addition for the training section
		for i in range(self.hiddenlayers+1):
			inputs = np.dot(self.weights[i], inputs)
			inputs += self.biases[i]
			inputs = sigmoid(inputs)
			layers.append(inputs)


		# the actual training 
		for i in range(self.hiddenlayers+1, 0, -1):

			# calculate the networks error
			error = targets - layers[i]

			# calculate the gradient of the network function
			gradient = sigmoid_p(layers[i])
			gradient *= error
			gradient *= self.learningRate

			# calculate the delta of the gradient
			delta = np.dot(gradient, layers[i-1].T)

			# adjust the weights and biases
			self.biases[i-1] += gradient
			self.weights[i-1] += delta

			# reset for the next layer
			t = self.weights[i-1].T
			t = np.dot(t, error)
			t += layers[i-1]
			targets = t



	def process(self, inputArray):
		if(len(inputArray) != self.inputNodes):
			raise Exception("the number of inputs must match the number of inputNodes")

		# feedforward algorithm
		inputs = np.array(inputArray)
		inputs = inputs.reshape(self.inputNodes, 1)
		for i in range(self.hiddenlayers+1):
			inputs = np.dot(self.weights[i], inputs)
			inputs += self.biases[i]
			inputs = sigmoid(inputs)
		return inputs

	# calculate the mean squared error
	def mse(self, inputArray, goalArray):
		guess = self.process(inputArray)
		return np.sum((goalArray-guess)**2)/len(goalArray)

	# calculate the root mean squared error
	def rmse(self, inputArray, goalArray):
		return sqrt((self.mse(inputArray, goalArray)/4))


