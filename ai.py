import numpy as np
import matplotlib.pyplot as plt
from perceptron import *

def truncate(x):
	return int(x*100)/100

def percent(x, total):
	return truncate((x/total)*100)

dataset = [[1,0,1],
			[0,1,1],
			[1,1,0],
			[0,0,0]]

brain = perceptron(2, 2, 100, 1)
epochs = 20000

brain.setLearningRate(.1)

errors = []
eps = []

# training loop
for i in range(epochs):
	index = np.random.randint(len(dataset))
	data = dataset[index]
	info = [data[0], data[1]]
	goal = [data[2]]
	brain.train(info, goal)
	errors.append(brain.mse(info, goal))
	eps.append(i)
	print("epoch: %d, error: %f, %g%% complete" % (i, brain.mse(info, goal), percent(i, epochs-1)))
	

# guessing loop
for i in range(4):
	data = dataset[i]
	info = [data[0], data[1]]
	goal = [data[2]]
	print(goal, brain.process(info), brain.mse(info, goal))