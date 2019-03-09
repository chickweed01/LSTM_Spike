using System.Collections.Generic;

namespace LSTM_UML.Nodes
{
    public abstract class AbstractNode
    {
        public abstract float Val { get; set; }
        public abstract List<Connector> InboundConnectors { get; set; }
        public abstract List<Connector> OutboundConnectors { get; set; }

        protected abstract void makeListOfConnectors();
    }
}
