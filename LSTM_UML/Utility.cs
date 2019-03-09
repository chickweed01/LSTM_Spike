using System;
using System.Collections.Generic;
using LSTM_UML.Gates;
using LSTM_UML.Nodes;

namespace LSTM_UML
{
    public class Utility
    {
        public static class GenericNodeFactory
        {
            private static IList<Type> _registeredTypes = new List<Type>();

            static GenericNodeFactory()
            {
                _registeredTypes.Add(typeof(InputNode));
                _registeredTypes.Add(typeof(ForgetGate));
                _registeredTypes.Add(typeof(InputGate));
                _registeredTypes.Add(typeof(OutputGate));
                _registeredTypes.Add(typeof(OutputNode));
                _registeredTypes.Add(typeof(BiasNode));
            }

            public static IList<T> CreateGenericNode<T>(int numberOfNodes)
            {
                var t = typeof(T);
                int index = _registeredTypes.IndexOf(t);
                var typeToCreate = _registeredTypes[index];
                IList<T> list = new List<T>();

                if (typeToCreate == typeof(InputNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        InputNode node = new InputNode();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(ForgetGate))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        ForgetGate node = new ForgetGate();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(InputGate))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        InputGate node = new InputGate();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(OutputGate))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        OutputGate node = new OutputGate();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(OutputNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        OutputNode node = new OutputNode();
                        list.Add((T)(object)node);
                    }
                }
                else if (typeToCreate == typeof(BiasNode))
                {
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        BiasNode node = new BiasNode();
                        list.Add((T)(object)node);
                    }
                }

                return list;
            }
        }

        public static void createConnectors(int numberOfConnectors, out List<Connector> connectors, 
                                            float[] weightsToAssign)
        {
            connectors = new List<Connector>();

            for (int index = 0; index < numberOfConnectors; index++)
            {
                connectors.Add(new Connector(weightsToAssign[index]));
            }
        }

        public static float Sigmoid(float x)
        {
            if (x < -10.0) return 0.0f;
            else if (x > 10.0) return 1.0f;
            return (float)(1.0 / (1.0 + Math.Exp(-x)));
        }

        public static float Tanh(float x)
        {
            if (x < -10.0) return -1.0f;
            else if (x > 10.0) return 1.0f;
            return (float)(Math.Tanh(x));
        }
    }
}