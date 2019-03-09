using LSTM_UML.Gates;
using LSTM_UML.Nodes;
using System;
using System.Collections.Generic;

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
