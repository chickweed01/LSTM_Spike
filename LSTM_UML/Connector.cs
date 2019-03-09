using System;
using System.Collections.Generic;
using System.Linq;
using LSTM_UML.Nodes;

namespace LSTM_UML
{
    public class Connector
    {
        public Connector(float weight = .05f)
        {
            Weight = weight;
        }
        public float Weight { get; set; }
        public AbstractNode FromGate { get; set; }
        public AbstractNode ToGate { get; set; }
    }
}
