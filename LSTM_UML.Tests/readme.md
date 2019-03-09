# Introduction
There are three main tests in the test project. They validate that the Forget, Input and Output Gates are constructed correctly using Node and Connector objects. They also validate that the summations at the gates are calculated correctly.

Before each test is executed a LSTM cell object is instantiated. The cell constructor knows how to create each gate which is modeled as a collection of neural network nodes and connectors. (See the LSTM_UML project for more details.)  The initial values for the connector weights are defined including those for the bias node connectors.

```
[TestMethod]
public void testSummationsAtForgetGate()
{            
    cell.SetupForgetGate(inputs, outputs, biases, W, U, b);

    cell.calcualteForgetGateValues();

    Assert.IsTrue(Math.Round(cell.ForgetGates[0].Val, 4) == 0.5523, "Expected fGate value of 0.5523, got {0}", cell.ForgetGates[0].Val);
    Assert.IsTrue(Math.Round(cell.ForgetGates[1].Val, 4) == 0.5695, "Expected fGate value of 0.5695, got {0}", cell.ForgetGates[1].Val);
    Assert.IsTrue(Math.Round(cell.ForgetGates[2].Val, 4) == 0.5866, "Expected fGate value of 0.5866, got {0}", cell.ForgetGates[2].Val);
}
        
[TestMethod]
public void testSummationsAtInputGate()
{
    cell.SetupInputGate(inputs, outputs, biases, W, U, b);

    cell.calculateInputGateValues();

    Assert.IsTrue(Math.Round(cell.InputGates[0].Val, 4) == 0.5523, "Expected iGate value of 0.5523, got {0}", cell.InputGates[0].Val);
    Assert.IsTrue(Math.Round(cell.InputGates[1].Val, 4) == 0.5695, "Expected iGate value of 0.5695, got {0}", cell.InputGates[1].Val);
    Assert.IsTrue(Math.Round(cell.InputGates[2].Val, 4) == 0.5866, "Expected iGate value of 0.5866, got {0}", cell.InputGates[2].Val);
}

[TestMethod]
public void testSummationsAtOutputGate()
{
    cell.SetupOutputGate(inputs, outputs, biases, W, U, b);

    cell.calculateOutputGateValues();

    Assert.IsTrue(Math.Round(cell.OutputGates[0].Val, 4) == 0.5523, "Expected oGate value of 0.5523, got {0}", cell.OutputGates[0].Val);
    Assert.IsTrue(Math.Round(cell.OutputGates[1].Val, 4) == 0.5695, "Expected oGate value of 0.5695, got {0}", cell.OutputGates[1].Val);
    Assert.IsTrue(Math.Round(cell.OutputGates[2].Val, 4) == 0.5866, "Expected oGate value of 0.5866, got {0}", cell.OutputGates[2].Val);
}
```        
