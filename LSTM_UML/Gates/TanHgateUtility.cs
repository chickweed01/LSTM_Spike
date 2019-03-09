using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LSTM_UML.Gates
{
    public static class TanHGateUtility
    {
        public static float[] calculateCellStateValues(List<TanhGate> TanhGates, List<ForgetGate> ForgetGates, 
                                                        List<InputGate> InputGates, float[] c_prev)
        {
            //initialize return array
            float[] c = new float[] { 0.0f, 0.0f, 0.0f };

            foreach (TanhGate thGate in TanhGates)
            {
                foreach (Connector connector in thGate.InboundConnectors)
                {
                    thGate.Val += connector.Weight * connector.FromGate.Val;
                }

                thGate.Val = Utility.Tanh(thGate.Val);
            }

            //ForgetGate values and the previous cell state values
            //perform the equivalent of element-wise multiplication of the
            int index = 0;
            foreach (ForgetGate fgate in ForgetGates)
            {
                c[index] = fgate.Val * c_prev[index];
                index++;
            }

            //TanhGate values and the InputGate
            //perform the equivalent of element-wise multiplication of the

            index = 0;
            foreach (TanhGate thGate in TanhGates)
            {
                c[index] += InputGates[index].Val * thGate.Val;
                index++;
            }

            return c;
        }
    }
}
