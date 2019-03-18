import numpy as np
import os
import random
from activations import sigmoid, sigmoid_p

# percentage and truncate functions used in displaying training completion
def truncate(x, level=100):
	return int(x*level)/level

def percent(x, total=100, level=100):
	return truncate((x/total)*100, level)

def largest_index(guess):
	biggest = 0
	biggestIndex = 0
	for i in range(len(guess)):
		if guess[i] > biggest:
			biggest = guess[i]
			biggestIndex = i
	return biggestIndex

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


	# automatically prepare the data from given training data and labels and run the train function epochs number of times
	def fit(self, input_data, labels):

		# error checking
		if len(input_data) > len(labels):
			raise Exception("Your have more data points than labels")

		if len(input_data) < len(labels):
			raise Exception("Your have more labels points than data points")

		datasize = len(input_data)
		for i in range(self.epochs):
			index = np.random.randint(datasize)
			info = input_data[index]
			goal = labels[index]
			self.train(info, goal)
			if (i+1) % 100 == 0:
				print("epoch: %d, error: %f, %g%% complete" % (i+1, self.mse(info, goal), percent(i, self.epochs-1, 100)))


	# train the network with backpropagation and stochastic gradient descent
	def train(self, inputArray, goalArray):
		inputs = np.array(inputArray)
		inputs = inputs.reshape(self.inputNodes, 1)
		targets = np.array(goalArray)
		targets = targets.reshape(self.outputNodes, 1)
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
			temp = self.weights[i-1].T
			temp = np.dot(temp, error)
			temp += layers[i-1]
			targets = temp


	def test(self, testdata, testlabels):
		acc = 0
		datalen = len(testdata)
		for i in range(100):
			data = testdata[i]
			label = testlabels[i]
			guess = self.process(data)
			guess = largest_index(guess)
			numberlabel = largest_index(label)
			error = self.mse(data, label)
			
			verdict = ""
			if guess == numberlabel:
				verdict = "correct"
				acc += 1
			else:
				verdict = "fail"
			print("answer: %d, guess: %i, error: %s, verdict: %s" % (numberlabel, 
														guess, 
														error, 
														verdict,))
		accuracy = percent(acc, 100)
		print("")
		print("accuracy: %i%%" % (accuracy))


	def process_all(self, inputs):
		for i in range(len(inputs)):
			data = inputs[i]
			guess = self.process(data)
			guess = largest_index(guess)
			print("input number: %d, guess: %f" % (i, guess))


	# feed the inputs forward through the network
	def process(self, inputArray):
		# if(len(inputArray) != self.inputNodes):
		# 	raise Exception("the number of inputs must match the number of inputNodes", len(inputArray))

		# feedforward algorithm
		inputs = np.array(inputArray)
		inputs = inputs.reshape(self.inputNodes, 1)
		for i in range(self.hiddenlayers+1):
			inputs = np.dot(self.weights[i], inputs)
			inputs += self.biases[i]
			inputs = sigmoid(inputs)
		return inputs


	# calculate the mean squared error, could be incorrect
	def mse(self, inputArray, goalArray):
		goal = np.array(goalArray)
		goal = goal.reshape(self.outputNodes, 1)
		guess = self.process(inputArray)
		return np.sum((goal-guess)**2)/len(goal)


	# calculate the root mean squared error, could be incorrect
	def rmse(self, inputArray, goalArray):
		return sqrt((self.mse(inputArray, goalArray)/4))


