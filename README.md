# simple-deep-neural-network
a simple deep neural network i made in python

## The Network
the network is a multiplayer deep neural with a variable number of input nodes, hidden layers, hidden nodes in those layers, output nodes, and training epochs.  
the network is object oriented, so you can bring it in to your program by importing the file and creating a variable for a neural network. The network takes in 5 variable for the constructor function. the number of inputs, layers, hiddens, outputs, and epochs in that order.
### training
training the network involves creating a for loop over the networks epochs, creating an list with a random entry from your training set and another list with the cooresponding label inside the loop, and using **.train** with the data and the label.  
![epochs](https://i.imgur.com/9JVRjhB.png)
![Training](https://i.imgur.com/DQp5Y3t.png)
### testing/predicting
testing the network and using it to predict data involves using the **.process** on your testing set and unknown set.
![guessing](https://i.imgur.com/iPCENrD.png)
