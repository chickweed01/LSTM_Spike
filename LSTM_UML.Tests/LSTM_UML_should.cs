using LSTM_UML.Gates;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace LSTM_UML.Tests
{
    [TestClass]
    public class LSTM_UML_should
    {
        int numberOInputs;
        int numberOfOutputs;

        [TestInitialize]
        public void Initialize()
        {
            numberOInputs = 2;
            numberOfOutputs = 3;
        }

        [TestMethod]
        public void testLSTM_CellCreation()
        {
            LSTM_Cell cell = new LSTM_Cell(2,3);
            List<ForgetGate> forgetGate = cell.ForgetGates;
            List<InputGate> inputGate = cell.InputGates;
            List<OutputGate> outputGate = cell.OutputGates;

            //forgetGate.ForgetGateVector = new float[] { 0.0f, 0.0f, 0.0f };

            //Assert.IsTrue(cell.ForgetGate.ForgetGateVector.Length == 3, "Expected 3 cells in forgetGate vector");
        }
    }
}
