using System.Collections.Generic;

namespace LSTM_UML.Nodes 
{
    public class OutputNode : AbstractNode
    {
        private float _val;

        public OutputNode(float val = 1.0f)
        {
            _val = val;

            makeListOfConnectors();
        }

        public override float Val { get { return _val; } set { _val = value; } }
        public override List<Connector> InboundConnectors { get; set; }
        public override List<Connector> OutboundConnectors { get; set; }
        protected override void makeListOfConnectors()
        {
            OutboundConnectors = new List<Connector>();
        }
    }
}