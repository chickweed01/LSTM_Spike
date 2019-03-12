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

So basically, the main components of an LSTM cell are unique neural networks (NN) connected in a sequence. Each node in a NN is modeled as a UML class having two lists of connectors - one for inbound connections and one for outbound connectons. Each pair of nodes is joined by a connector class having a weight. The connectors themselves are assigned a single node at each end representing a "from node" and a "to node".

The UML representation of an LSTM cell is mainly a set of four collections of nodes - one for each gate and the TanH layer. The complete class diagram is shown below:

## Class Diagram
![Image](/LSTM_UML/images/LSTM_Cell_Class_Diagram.jpg?raw=true "LSTM Cell Class Diagram")

## C# Code

The main console application code is shown below. It was written using the VSCode IDE.
```
namespace LSTM_UML
{
    class Program
    {
        static void Main(string[] args)
        {
            int numberOfInputs = 2;
            int numberOfOutputs = 3;
            LSTM_Cell cell = new LSTM_Cell(numberOfInputs, numberOfOutputs);
            List<Connector> connectors;

            List<InputNode> inputs = Utility.GenericNodeFactory.CreateGenericNode<InputNode>(numberOfInputs) as List<InputNode>;
            List<OutputNode> outputs = Utility.GenericNodeFactory.CreateGenericNode<OutputNode>(numberOfOutputs) as List<OutputNode>;
            List<BiasNode> biases = Utility.GenericNodeFactory.CreateGenericNode<BiasNode>(numberOfOutputs) as List<BiasNode>;

            Utility.createConnectors(2, out connectors, new float[] { .01f, .01f });

            //reference values
            float[] xt = new float[] { 1.0f, 2.0f };

            //set the values of the inputs
            inputs[0].Val = xt[0];
            inputs[1].Val = xt[1];

            /* set the values of the outputs these have the same values 
             * as the h_prev vector items which are 0.0 initially */
            outputs[0].Val = 0.0f;
            outputs[1].Val = 0.0f;
            outputs[2].Val = 0.0f;

            //initial input weights
            float[] W = new float[] { 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f };
            float[] U = new float[] { 0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f };
            float[] b = new float[] { 0.16f, 0.17f, 0.18f };
            
            float[] c = new float[] { 0.0f, 0.0f, 0.0f };
            float[] c_prev = new float[] { 0.0f, 0.0f, 0.0f };
            float[] h = new float[] { 0.0f, 0.0f, 0.0f };
            float[] h_prev = new float[] { 0.0f, 0.0f, 0.0f };

            // 1) calculate Forget Gate sums
            cell.SetupForgetGate(inputs, outputs, biases, W, U, b);
            cell.calcualteForgetGateValues();
            Console.WriteLine("Initial Forget Gate values: {0}, {1}, {2}", cell.ForgetGates[0].Val, cell.ForgetGates[1].Val, cell.ForgetGates[2].Val);

            // 2) calculate Input Gate sums
            cell.SetupInputGate(inputs, outputs, biases, W, U, b);
            cell.calculateInputGateValues();

            // 3) calculate outout gate sums
            cell.SetupOutputGate(inputs, outputs, biases, W, U, b);
            cell.calculateOutputGateValues();

            // 4) calculate current cell state
            cell.SetupCurrentCellState(inputs, outputs, biases, W, U, b, c_prev);
            //c = cell.calculateCellStateValues(c_prev);
            c = TanHGateUtility.calculateCellStateValues(cell.TanhGates, cell.ForgetGates, cell.InputGates,  c_prev);

            c_prev = c;

            // 5) calculate current outputs
            h = cell.calculateOutputVector(c);

            h_prev = h;

            //set the outputs to the previous set of outputs
            outputs[0].Val = h_prev[0];
            outputs[1].Val = h_prev[1];
            outputs[2].Val = h_prev[2];

            Console.WriteLine("=====");
            Console.WriteLine("\nSending input = (3.0, 4.0) to LSTM \n");

            //send in the next set of inputs to see
            //how the outputs and cell state change
            xt = new float[] { 3.0f, 4.0f };
            //set the values of the inputs
            inputs[0].Val = xt[0];
            inputs[1].Val = xt[1];

            //reset the gate values to 0.0
            cell.resetGateValues();

            // 1) calculate Forget Gate sums
            cell.calcualteForgetGateValues();
            Console.WriteLine("Forget Gate values: {0}, {1}, {2}", cell.ForgetGates[0].Val, cell.ForgetGates[1].Val, cell.ForgetGates[2].Val);

            // 2) calculate Input Gate sums
            cell.calculateInputGateValues();
            Console.WriteLine("Input Gate values: {0}, {1}, {2}", cell.InputGates[0].Val, cell.InputGates[1].Val, cell.InputGates[2].Val);

            // 3) calculate outout gate sums
            cell.calculateOutputGateValues();
            Console.WriteLine("Output Gate values: {0}, {1}, {2}", cell.OutputGates[0].Val, cell.OutputGates[1].Val, cell.OutputGates[2].Val);

            // 4) calculate current cell state
            //c = cell.calculateCellStateValues(c_prev);
            c = TanHGateUtility.calculateCellStateValues(cell.TanhGates, cell.ForgetGates, cell.InputGates, c_prev);
            Console.WriteLine("Cell state is: {0}, {1}, {2}", c[0], c[1], c[2]);

            c_prev = c;

            // 5) calculate current outputs
            h = cell.calculateOutputVector(c);
            Console.WriteLine("New Output is: {0}, {1}, {2}", h[0], h[1], h[2]);

            h_prev = h;

            Console.WriteLine("End LSTM demo ");
            Console.ReadLine();
        } //Main
    }
}
```

The LSTM_Cell constructor instantiates the set of gates:
```
//constructor
public LSTM_Cell(int numInputs, int numOutputs)
{
    _numInputs = numInputs;
    _numOutputs = numOutputs;
    ForgetGates = new List<ForgetGate>(numOutputs);
    InputGates = new List<InputGate>(numOutputs);
    OutputGates = new List<OutputGate>(numOutputs);
    TanhGates = new List<TanhGate>(numInputs);

    addGateInstances(numOutputs);
}
```
The LSTM_Cell's "addGateInstances" method instantiates the gate nodes for each gate and the cell state nodes.
```
private void addGateInstances(int numberOfGates)
{
    //add ForgetGate instances to cell's list
    for (int index = 0; index < numberOfGates; index++)
    {
        ForgetGate fGate = new ForgetGate();
        ForgetGates.Add(fGate);
    }

    //add InputGate instances to cell's list
    for (int index = 0; index < numberOfGates; index++)
    {
        InputGate iGate = new InputGate();
        InputGates.Add(iGate);
    }

    //add OutputGate instances to cell's list
    for (int index = 0; index < numberOfGates; index++)
    {
        OutputGate oGate = new OutputGate();
        OutputGates.Add(oGate);
    }

    //add TanhGate instances to cell's list
    for (int index = 0; index < numberOfGates; index++)
    {
        TanhGate thGate = new TanhGate();
        TanhGates.Add(thGate);
    }
}
```

The Main program calls the methods that create the neural networks for each gate and the cell state. Those methods are:
* SetupForgetGate
* SetupInputGate
* SetupOutputGate
* SetupCurrentCellState

After the neural networks are constructed for each gate the summation calculations that occur at each gate are performed. The methods called are:
* calcualteForgetGateValues
* calculateInputGateValues
* calculateOutputGateValues
* calculateCellStateValues

Once the gate calculations are performed the cell's output is calculated by calling the "calculateOutputVector" method. A Utility class is included in the source code listing that contains methods for:
* creating nodes
* creating connectors
* calculating the LogSigmoid and Tanh

## License
[MIT](https://choosealicense.com/licenses/mit/)
