import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_p(x):
	return x * (1-x)

def tanh(x):
	return np.tanh(x)

def tanh_p(x):
	return 1 - x * x

class perceptron:
	def __init__(self, inputs, layers, hidden, output):
		self.inputNodes = int(inputs)
		self.hiddenlayers = int(layers)
		self.hiddenNodes = int(hidden)
		self.outputNodes = int(output)

		self.learningRate = .1

		self.weights = []

		for i in range(self.hiddenlayers+1):
			if i == 0:
				self.weights.append(np.random.rand(self.hiddenNodes, self.inputNodes))
			elif i == self.hiddenlayers:
				self.weights.append(np.random.rand(self.outputNodes, self.hiddenNodes))
			else:
				self.weights.append(np.random.rand(self.hiddenNodes, self.hiddenNodes))

		self.biases = []

		for i in range(self.hiddenlayers+1):
			if i == self.hiddenlayers:
				self.biases.append(np.random.rand(self.outputNodes,1))
			else:
				self.biases.append(np.random.rand(self.hiddenNodes, 1))

	def setLearningRate(self, x):
		self.learningRate = x

	def getLearningRate(self):
		return self.learningRate

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
			error = (targets - layers[i])

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
		return pow((guess-goalArray), 2)

	# calculate the root mean squared error
	def rmse(self, inputArray, goalArray):
		return sqrt((self.mse(inputArray, goalArray)/4))


