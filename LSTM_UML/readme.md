# LSTM via UML
> An exploration of Long Short-term Memory Cells using a UML-based approach

## Introduction
In my quest to have a better understanding if how Long Short-Term Memory (LSTM) cells function I decided to model a cell using UML and implement it in C#. (I'll assume that you are familar with the use of LSTM cells in recurrent neural networks, so I won't cover it here. This [blog](https://bit.ly/1iaBaLH) provides a great explanation.)  The question then became how to validate that my implementation was accurate. Fortunately, I came across an [article](https://bit.ly/2NMdn5n) in MSDN Magazine by James McCaffrey of Microsoft Research. In his article Dr. McCaffrey describes an implemention of a LSTM cell written in C#. With Dr McCaffrey's program in hand to be used as a benchmark I set out to create a model of an LSTM cell and its components expressed as UML classes.

## The Model

The basic function of an LSTM cell is to determine how much information about the previous cell state and previous output to persist in the cell's current data. That determination is made using a series of "gates". A diagram of a typical cell with its gates is shown below:

![Image](/LSTM_UML/images/LSTMCell.jpg?raw=true "LSTM Cell")

One approach to modeling an LSTM cell is to treat each gate as its own neural network. The image below shows a neural net for the Forget Gate. 

![Image](/LSTM_UML/images/ForgetGateSummations.jpg?raw=true "Forget Gate")

The equation for the summations at the Forget Gate is:

```
f(t) = sigmoid(Wx(t) + Uh(t-1) + b) where:
* x(t) = current input values
* W = input weights for the Forget Gate
* U = output weights for the Forget Gate
* h(t-1) = previous outputs
* b = Forget Gate biases node weights
```

The Input and Output Gates are modeled in the same way and are also calculated in a similar way. The unique part of a gate's calculations is the set of weights. Each gate has its own set of weights. The TanH layer also has its own neural network with its own set of weights.

So basically, the main components of an LSTM cell are unique neural networks (NN) connected in a sequence. Each node in a NN is a modeled as a UML class having two lists or connectors - one for inbound connections and one for outbound connectons. Each pair of nodes is joined by a connector class having a weight. The connectors themselves are assigned a single node at each end representing a "from node" and a "to node".

The UML representation of an LSTM cell is mainly a set of four collections of nodes - one for each gate and the TanH layer. The complete class diagram is shown below:

## Class Diagram
![Image](/LSTM_UML/images/LSTM_Cell_Class_Diagram.jpg?raw=true "LSTM Cell Class Diagram")

## C# Code

The main console application code is shown below. It was written using the VSCode IDE

## License
[MIT](https://choosealicense.com/licenses/mit/)
