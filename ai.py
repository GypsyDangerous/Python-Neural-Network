def percent(x, total):
	return truncate((x/total)*100)

dataset = [[1,0,1],
	[0,1,1],
	[1,1,0],
	[0,0,0]]

brain = perceptron(2, 1, 50, 1, 20000)
epochs = brain.getEpochs()

brain.setLearningRate(.1)
datasize = len(dataset)

errors = []
eps = []

# training loop
for i in range(epochs):
	index = np.random.randint(datasize)
	data = dataset[index]
	info = [data[0], data[1]]
	goal = [data[2]]
	brain.train(info, goal)
	errors.append(brain.mse(info, goal))
	print("epoch: %d, error: %f, %g%% complete" % (i, brain.mse(info, goal), percent(i, epochs-1)))
	
print("")

plt.plot(errors)

# guessing loop
for i in range(datasize):
	data = dataset[i]
	info = [data[0], data[1]]
	goal = [data[2]]
	print("answer: %d, guess: %f, error: %s" % (goal[0], brain.process(info)[0], brain.mse(info, goal)))

plt.show()
