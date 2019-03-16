import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_p(x):
	return x * (1-x)

def tanh(x):
	return np.tanh(x)

def tanh_p(x):
	return 1 - x * x

def relu(x):
	return np.maximum(0, x)

def relu_p(x):
	return float(x > 0)