using System.Collections.Generic;

namespace LSTM_UML.Nodes
{
    public class InputNode : AbstractNode
    {
        private float _val;
        
        public InputNode(float val = 0.0f)
        {
            _val = val;

            makeListOfConnectors();
        }
        public override float Val { get {return _val; } set {_val = value; } }
        public override List<Connector> OutboundConnectors { get; set; }
        public override List<Connector> InboundConnectors { get; set; }

        protected override void makeListOfConnectors()
        {            
            OutboundConnectors = new List<Connector>();      
        }
    }
}