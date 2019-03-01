using LSTM_UML.Gates;
using System.Collections.Generic;

namespace LSTM_UML
{
    public class LSTM_Cell
    {
        public LSTM_Cell(int numInputs, int numOutputs)
        {
            ForgetGates = new List<ForgetGate>(numOutputs);
            InputGates = new List<InputGate>(numOutputs);
            OutputGates = new List<OutputGate>(numOutputs);
        }

        public float[] State { get; set; }
        public float[] Output { get; set; }
        public List<ForgetGate> ForgetGates { get; set; }
        public List<InputGate> InputGates { get; set; }
        public List<OutputGate> OutputGates { get; set; }
    }
}
