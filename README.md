# simple-deep-neural-network
a simple deep neural network I made in python only using numpy.

## The Network
the network is a multiplayer deep neural with a variable number of input nodes, hidden layers, hidden nodes in those layers, output nodes, and training epochs. It is a deep neural network but it is probably not the most efficient code.
the network is object oriented, so you can bring it in to your program by importing the file and creating a variable for a neural network. The network takes in 5 variable for the constructor function. the number of inputs, layers, hiddens, outputs, and epochs in that order.
### functions
* setLearningRate - sets the networks learning rate.
* getLearningRate - returns the networks learning rate.
* setEpochs - sets the networks training epochs.
* getEpochs - returns the networks training epochs.
* train - uses backpropagation and stochastic gradient descent to train the network.
* process - uses feedforward to guess the output of an input.
### training
training the network involves creating a for loop over the networks epochs, creating an list with a random entry from your training set and another list with the cooresponding label inside the loop, and using the **.train** function with the data and the label.  
![epochs](https://i.imgur.com/9JVRjhB.png)
![Training](https://i.imgur.com/DQp5Y3t.png)
### testing/predicting
testing the network and using it to predict data involves using the **.process** function on your testing set and unknown set.
![guessing](https://i.imgur.com/iPCENrD.png)
## the current test program
currently the network is being tested on XOR. It outputs every epoch for the training cycle and then when it is finished training it displays the networks guess and graphs the error.
![guesses](https://i.imgur.com/4mbQmLi.png)   
![error](https://i.imgur.com/LFpAf3Q.png)
### to do
- [ ] fix mse and rmse - currently they are my best guess as to how those function work, but they could very well be wrong
- [ ] refactor the code
- [ ] add more documentation with commments inside the code
- [ ] add functionality for easy switching between different activation functions
- [ ] add more activation functions

### credits and links to learn more
[The Coding Train](https://www.youtube.com/user/shiffman) introduced me to [neural networks](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6Y7MdSCaIfsxc561QI0U0Tb).  
[Siraj Rival](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) has helped me greatly with learning about ml/neural networks.  
[A java neural network class](https://github.com/Fir3will/Java-Neural-Network) that I used to better understand the code behind the math.  
[online book](http://neuralnetworksanddeeplearning.com/) that helped me understand the math.  
[3blue1brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) on [neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), on [calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr), on [linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).  
[Khan Academy](https://www.khanacademy.org/).
