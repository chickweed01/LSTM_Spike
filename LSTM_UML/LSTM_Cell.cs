using LSTM_UML.Gates;
using LSTM_UML.Nodes;
using System;
using System.Collections.Generic;

namespace LSTM_UML
{
    public class LSTM_Cell
    {
        private int _numInputs;
        private int _numOutputs;

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

        public List<TanhGate> TanhGates { get; set; }        
        public List<ForgetGate> ForgetGates { get; set; }
        public List<InputGate> InputGates { get; set; }
        public List<OutputGate> OutputGates { get; set; }

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

        public void SetupForgetGate(List<InputNode> inputs, List<OutputNode> outputs,
                                                List<BiasNode> biases, float[] inputGateWeights, 
                                                float[] outputGateWeights, float[] biasWeights)
        {
            int iConnectorCounter = 0;
            List<Connector> inputsToForgetGateConnectors;
            Utility.createConnectors(6, out inputsToForgetGateConnectors, inputGateWeights);

            //hook-up the connectors
            //connect from an input node to a ForgetGate
            for (int j = 0; j < ForgetGates.Count; j++)
            {
                for (int i = 0; i < _numInputs; i++)
                {
                    inputs[i].OutboundConnectors.Add(inputsToForgetGateConnectors[iConnectorCounter]);
                    inputsToForgetGateConnectors[iConnectorCounter].FromGate = inputs[i];
                    ForgetGates[j].InboundConnectors.Add(inputsToForgetGateConnectors[iConnectorCounter]);
                    inputsToForgetGateConnectors[iConnectorCounter].ToGate = ForgetGates[j];

                    iConnectorCounter++;
                }
            }

            //connect from an output node to a ForgetGate nodes
            //the output nodes feed into the forget gate
            iConnectorCounter = 0;
            List<Connector> outputsToForgetGateConnectors;
            Utility.createConnectors(9, out outputsToForgetGateConnectors, outputGateWeights);

            for (int j = 0; j < ForgetGates.Count; j++)
            {
                for (int i = 0; i < _numOutputs; i++)
                {
                    outputs[i].OutboundConnectors.Add(outputsToForgetGateConnectors[iConnectorCounter]);
                    outputsToForgetGateConnectors[iConnectorCounter].FromGate = outputs[i];
                    ForgetGates[j].InboundConnectors.Add(outputsToForgetGateConnectors[iConnectorCounter]);
                    outputsToForgetGateConnectors[iConnectorCounter].ToGate = ForgetGates[j];

                    iConnectorCounter++;
                }
            }

            //connect the bias nodes to the ForgetGate nodes
            iConnectorCounter = 0;
            List<Connector> biasToForgetGateConnectors;
            Utility.createConnectors(3, out biasToForgetGateConnectors, biasWeights);
            for (int i = 0; i < biasWeights.Length; i++)
            {
                biases[i].OutboundConnectors.Add(biasToForgetGateConnectors[iConnectorCounter]);
                biasToForgetGateConnectors[iConnectorCounter].FromGate = biases[i];
                ForgetGates[i].InboundConnectors.Add(biasToForgetGateConnectors[iConnectorCounter]);
                biasToForgetGateConnectors[iConnectorCounter].ToGate = ForgetGates[i];

                iConnectorCounter++;
            }
        }

        public void calcualteForgetGateValues()
        {
            foreach (ForgetGate fGate in ForgetGates)
            {
                foreach (Connector connector in fGate.InboundConnectors)
                {
                    fGate.Val += connector.Weight * connector.FromGate.Val;
                }

                fGate.Val = Utility.Sigmoid(fGate.Val);
            }
        }

        public void SetupInputGate(List<InputNode> inputs, List<OutputNode> outputs,
                                                List<BiasNode> biases, float[] inputGateWeights,
                                                float[] outputGateWeights, float[] biasWeights)
        {
            //connect from an input node to a InputGate
            int iConnectorCounter = 0;
            List<Connector> inputsToInputGateConnectors;
            Utility.createConnectors(6, out inputsToInputGateConnectors, inputGateWeights);
            for (int j = 0; j < InputGates.Count; j++)
            {
                for (int i = 0; i < _numInputs; i++)
                {
                    inputs[i].OutboundConnectors.Add(inputsToInputGateConnectors[iConnectorCounter]);
                    inputsToInputGateConnectors[iConnectorCounter].FromGate = inputs[i];
                    InputGates[j].InboundConnectors.Add(inputsToInputGateConnectors[iConnectorCounter]);
                    inputsToInputGateConnectors[iConnectorCounter].ToGate = InputGates[j];

                    iConnectorCounter++;
                }
            }

            //connect from an output node to a InputGate nodes
            //the output nodes feed into the InputGate
            iConnectorCounter = 0;
            List<Connector> outputsToInputGateConnectors;
            Utility.createConnectors(9, out outputsToInputGateConnectors, outputGateWeights);
            for (int j = 0; j < InputGates.Count; j++)
            {
                for (int i = 0; i < _numOutputs; i++)
                {
                    outputs[i].OutboundConnectors.Add(outputsToInputGateConnectors[iConnectorCounter]);
                    outputsToInputGateConnectors[iConnectorCounter].FromGate = outputs[i];
                    InputGates[j].InboundConnectors.Add(outputsToInputGateConnectors[iConnectorCounter]);
                    outputsToInputGateConnectors[iConnectorCounter].ToGate = InputGates[j];

                    iConnectorCounter++;
                }
            }

            //connect the bias nodes to the InputGate nodes
            iConnectorCounter = 0;
            List<Connector> biasToInputGateConnectors;
            Utility.createConnectors(3, out biasToInputGateConnectors, biasWeights);
            for (int i = 0; i < biasWeights.Length; i++)
            {
                biases[i].OutboundConnectors.Add(biasToInputGateConnectors[iConnectorCounter]);
                biasToInputGateConnectors[iConnectorCounter].FromGate = biases[i];
                InputGates[i].InboundConnectors.Add(biasToInputGateConnectors[iConnectorCounter]);
                biasToInputGateConnectors[iConnectorCounter].ToGate = InputGates[i];

                iConnectorCounter++;
            }
        }

        public void calculateInputGateValues()
        {
            foreach (InputGate iGate in InputGates)
            {
                foreach (Connector connector in iGate.InboundConnectors)
                {
                    iGate.Val += connector.Weight * connector.FromGate.Val;
                }

                iGate.Val = Utility.Sigmoid(iGate.Val);
            }
        }

        public void SetupOutputGate(List<InputNode> inputs, List<OutputNode> outputs,
                                                List<BiasNode> biases, float[] inputGateWeights,
                                                float[] outputGateWeights, float[] biasWeights)
        {
            int iConnectorCounter = 0;
            List<Connector> inputsToOutputGateConnectors;
            Utility.createConnectors(6, out inputsToOutputGateConnectors, inputGateWeights);

            //hook-up the connectors
            //connect from an input node to a OutputGate
            iConnectorCounter = 0;
            for (int j = 0; j < OutputGates.Count; j++)
            {
                for (int i = 0; i < _numInputs; i++)
                {
                    inputs[i].OutboundConnectors.Add(inputsToOutputGateConnectors[iConnectorCounter]);
                    inputsToOutputGateConnectors[iConnectorCounter].FromGate = inputs[i];
                    OutputGates[j].InboundConnectors.Add(inputsToOutputGateConnectors[iConnectorCounter]);
                    inputsToOutputGateConnectors[iConnectorCounter].ToGate = OutputGates[j];

                    iConnectorCounter++;
                }
            }

            //connect from an output node to a OutputGate nodes
            //the output nodes feed into the OutputGate
            iConnectorCounter = 0;
            List<Connector> outputsToOutputGateConnectors;
            Utility.createConnectors(9, out outputsToOutputGateConnectors, outputGateWeights);
            for (int j = 0; j < OutputGates.Count; j++)
            {
                for (int i = 0; i < _numOutputs; i++)
                {
                    outputs[i].OutboundConnectors.Add(outputsToOutputGateConnectors[iConnectorCounter]);
                    outputsToOutputGateConnectors[iConnectorCounter].FromGate = outputs[i];
                    OutputGates[j].InboundConnectors.Add(outputsToOutputGateConnectors[iConnectorCounter]);
                    outputsToOutputGateConnectors[iConnectorCounter].ToGate = OutputGates[j];

                    iConnectorCounter++;
                }
            }

            //connect the bias nodes to the InputGate nodes
            iConnectorCounter = 0;
            List<Connector> biasToOutputGateConnectors;
            Utility.createConnectors(3, out biasToOutputGateConnectors, biasWeights);
            for (int i = 0; i < biasWeights.Length; i++)
            {
                biases[i].OutboundConnectors.Add(biasToOutputGateConnectors[iConnectorCounter]);
                biasToOutputGateConnectors[iConnectorCounter].FromGate = biases[i];
                OutputGates[i].InboundConnectors.Add(biasToOutputGateConnectors[iConnectorCounter]);
                biasToOutputGateConnectors[iConnectorCounter].ToGate = OutputGates[i];

                iConnectorCounter++;
            }

            //calculate sums at the OutputGate elements and apply log sig activation
            //and compare to expected values of the output gate values: 0.5523, 0.5695, 0.5866
            //loop through the OutputGate list and sum all of the weight * input values.
            //calculateOutputGateValues();
        }

        public void calculateOutputGateValues()
        {
            foreach (OutputGate oGate in OutputGates)
            {
                foreach (Connector connector in oGate.InboundConnectors)
                {
                    oGate.Val += connector.Weight * connector.FromGate.Val;
                }

                oGate.Val = Utility.Sigmoid(oGate.Val);
            }
        }

        public void SetupCurrentCellState(List<InputNode> inputs, List<OutputNode> outputs,
                                                List<BiasNode> biases, float[] inputGateWeights,
                                                float[] outputGateWeights, float[] biasWeights, float[] c_prev)
        {
            int iConnectorCounter = 0;
            //float[] c = new float[] { 0.0f, 0.0f, 0.0f };
            List<Connector> inputsToTanhGateConnectors;
            Utility.createConnectors(6, out inputsToTanhGateConnectors, inputGateWeights);

            //hook-up the connectors
            //connect from an input node to a OutputGate
            iConnectorCounter = 0;
            for (int j = 0; j < TanhGates.Count; j++)
            {
                for (int i = 0; i < _numInputs; i++)
                {
                    inputs[i].OutboundConnectors.Add(inputsToTanhGateConnectors[iConnectorCounter]);
                    inputsToTanhGateConnectors[iConnectorCounter].FromGate = inputs[i];
                    TanhGates[j].InboundConnectors.Add(inputsToTanhGateConnectors[iConnectorCounter]);
                    inputsToTanhGateConnectors[iConnectorCounter].ToGate = TanhGates[j];

                    iConnectorCounter++;
                }
            }

            //connect from an output node to a OutputGate nodes
            //the output nodes feed into the OutputGate
            iConnectorCounter = 0;
            List<Connector> outputsToTanhGateConnectors;
            Utility.createConnectors(9, out outputsToTanhGateConnectors, outputGateWeights);
            for (int j = 0; j < OutputGates.Count; j++)
            {
                for (int i = 0; i < _numOutputs; i++)
                {
                    outputs[i].OutboundConnectors.Add(outputsToTanhGateConnectors[iConnectorCounter]);
                    outputsToTanhGateConnectors[iConnectorCounter].FromGate = outputs[i];
                    TanhGates[j].InboundConnectors.Add(outputsToTanhGateConnectors[iConnectorCounter]);
                    outputsToTanhGateConnectors[iConnectorCounter].ToGate = TanhGates[j];

                    iConnectorCounter++;
                }
            }

            //connect the bias nodes to the InputGate nodes
            iConnectorCounter = 0;
            List<Connector> biasToTanhGateConnectors;
            Utility.createConnectors(3, out biasToTanhGateConnectors, biasWeights);
            for (int i = 0; i < biasWeights.Length; i++)
            {
                biases[i].OutboundConnectors.Add(biasToTanhGateConnectors[iConnectorCounter]);
                biasToTanhGateConnectors[iConnectorCounter].FromGate = biases[i];
                TanhGates[i].InboundConnectors.Add(biasToTanhGateConnectors[iConnectorCounter]);
                biasToTanhGateConnectors[iConnectorCounter].ToGate = TanhGates[i];
                iConnectorCounter++;
            }
        }

        public void resetGateValues()
        {
            foreach(ForgetGate gate in ForgetGates)
            {
                gate.Val = 0.0f;
            }

            foreach (InputGate gate in InputGates)
            {
                gate.Val = 0.0f;
            }

            foreach (OutputGate gate in OutputGates)
            {
                gate.Val = 0.0f;
            }

            foreach (TanhGate gate in TanhGates)
            {
                gate.Val = 0.0f;
            }
        }

        public float[] calculateOutputVector(float[] c)
        {
            int index = 0;
            float[] h = new float[] { 0.0f, 0.0f, 0.0f };

            //perform element wise multiplication of the
            //Output Gate values and the tanh of the current cell state values

            foreach (float item in c)
            {
                h[index] = OutputGates[index].Val * Utility.Tanh(item);
                index++;
            }
            return h;
        }
    }
}
