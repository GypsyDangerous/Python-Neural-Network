import numpy as np
from perceptron import *
import matplotlib.pyplot as plt


# simple XOR dataset
dataset = [[[1,0], [1]],
			[[0,1], [1]],
			[[1,1], [0]],
			[[0,0], [0]]]  
datasize = len(dataset)

# initialize the Neural Network
brain = perceptron(2, 1, 4, 1, 20000)

brain.setLearningRate(.1)


# training loop
brain.fit(dataset)
		
print("")

# guessing loop
for i in range(datasize):
	data = dataset[i]
	data_len = len(data)-1
	info = data[0]
	goal = data[1]
	guess = brain.process(info)
	error = brain.mse(info, goal)
	print("answer: %d, guess: %f, error: %s" % (goal[0], 
						    guess, 
						    error))
