import numpy as np
from perceptron import *

# simple XOR dataset
data = [[1, 0], [0, 1], [1, 1], [0, 0]]
labels = [[1], [1], [0], [0]]

datasize = len(data)
# initialize the Neural Network
brain = perceptron(2, 1, 10, 1, 200000)
epochs = brain.getEpochs()
brain.setLearningRate(.1)

# training loop
brain.fit(data, labels)
		
print("")

# guessing loop
for i in range(datasize):
	info = data[i]
	goal = labels[i]
	guess = brain.process(info)
	error = brain.mse(info, goal)
	print("answer: %d, guess: %f, error: %s" % (goal[0], 
						    guess, 
						    error))
