using LSTM_UML.Nodes;
using System.Collections.Generic;

namespace LSTM_UML.Gates
{
    public class InputGate: AbstractNode
    {
        private float _val;
        
        public InputGate(float val = 0.0f)
        {
            _val = val;

            makeListOfConnectors();
    }
        public override float Val { get {return _val; } set {_val = value; } }
        public override List<Connector> OutboundConnectors { get; set; }
        public override List<Connector> InboundConnectors { get; set; }

        protected override void makeListOfConnectors()
        {
            InboundConnectors = new List<Connector>();
        }
    }
}