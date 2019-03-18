# Python-Neural-Network

A simple deep neural network I made in python on my quest to understand machine learning and neural networks.


## Files
* ``ai.py`` - testing the neural network
* ``perceptron.py`` - the actual neural network
* ``activations.py`` - different activation functions for the network


## The Network

The network is a simple deep neural network with 5 variable number of input nodes, hidden layers, hidden nodes in those layers, output nodes, and training epochs. Though it is a deep neural network it is not the most efficient code.
the network is object oriented, so you can bring it in to your program by importing the file ``perceptron.py`` and creating a variable for a neural network. The network takes in 5 variables for the constructor function. the number of inputs, layers, hiddens, outputs, and epochs, in that order.

### Functions

* setLearningRate - sets the networks learning rate.
* getLearningRate - returns the networks learning rate.
* setEpochs - sets the networks training epochs.
* getEpochs - returns the networks training epochs.
* fit - run the train function on the training data ``self.epochs`` number of times
* train - uses backpropagation and stochastic gradient descent to train the network.
* process - uses feedforward to guess the output of an input.
* process_all - takes in a collection of unknown data and outputs the guess for each.
* test - takes in training data and training labels, runs the data through the network, compares to the label and gives you the networks accuracy.


### Training

Training the network involves creating a for loop over the networks epochs, creating an list with a random entry from your training set and another list with the cooresponding label inside the loop, and using the ``.train`` function with the data and the label. or simply calling ``.fit`` on your training data.  
* train function  
![epochs](https://i.imgur.com/9JVRjhB.png)
![Training](https://i.imgur.com/DQp5Y3t.png)  
* fit function  
![fit](https://i.imgur.com/6oVoRrw.png)

### Testing/Predicting

To test the network, simply use ``.test`` on you testdata and testlabels, To use it for predictions, either use the ``.process`` function on a single piece of unknown data or ``.process_all`` on a bunch of unknowns.
![guessing](https://i.imgur.com/iPCENrD.png)

## The current test program

Currently the network is being tested on XOR. It outputs every epoch for the training cycle and then when it is finished training it displays the networks guess and graphs the error. you can see the network run on MNIST data [here](https://github.com/GypsyDangerous/MNIST-digit-classifier/blob/master/README.md)
![guesses](https://i.imgur.com/4mbQmLi.png)   
![error](https://i.imgur.com/LFpAf3Q.png)  

## To Do List

- [ ] fix mse and rmse - currently they are my best guess as to how those function work, but they could very well be wrong (see [#2](https://github.com/GypsyDangerous/Python-Neural-Network/issues/2))
- [ ] add other loss functions like cross entropy
- [ ] refactor the code
- [ ] add more documentation with commments inside the code
- [ ] add functionality for easy switching between different activation functions
- [ ] add more activation functions
- [ ] add functionality for mini batch training
- [x] run this network on MNIST or similiar dataset [here](https://github.com/GypsyDangerous/MNIST-digit-classifier)
- [ ] add recurent neural network and convolutional neural network

## Credits and links to learn more

* [The Coding Train](https://www.youtube.com/user/shiffman) introduced me to [neural networks](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6Y7MdSCaIfsxc561QI0U0Tb).  
* [Siraj Rival](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) has helped me greatly with learning about ml/neural networks.  
* [A java neural network class](https://github.com/Fir3will/Java-Neural-Network) that I used to better understand the code behind the math.  
* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) book that helped me understand the math.  
* [3blue1brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) on [neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), on [calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr), on [linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).  
* [Khan Academy](https://www.khanacademy.org/).
* [Brilliant](https://brilliant.org/)
