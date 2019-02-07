using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    // A class to represent a layer in the network along with it's node values and weight.
    class Layer
    {

        // Node Bias of nodes in this layer.
        public double[] nodeBias;

        // The array that stores the current node value/output/activation for each node in this layer.
        public double[] nodeOutput;

        // Weight Array of weight values from this layer to next.
        public double[,] weight;

        // Used in backpropigation step to speed up computation.
        public double[,] weightErrorDerivative;
        public double[] activationDerivative;
        public double[] ENArray;

        // Number of nodes in this layer.
        public int nodeNumber;

        // Number of Nodes in the next layer.
        public int nextLayerNodeNumber;

        Random rnd;

        // Initialisation of the Layer.
        public Layer(int nn, int nlnn)
        {
            nodeNumber = nn;
            nextLayerNodeNumber = nlnn;
            weight = new double[nodeNumber, nextLayerNodeNumber];
            weightErrorDerivative = new double[nodeNumber, nextLayerNodeNumber];
            weightErrorDerivative = new double[nodeNumber, nextLayerNodeNumber];
            nodeOutput = new double[nodeNumber];
            ENArray = new double[nodeNumber];
            nodeBias = new double[nodeNumber];
            activationDerivative = new double[nodeNumber];
            rnd = new Random();
            for (int i = 0; i < nodeBias.Length; i++)
            {
                nodeBias[i] = rnd.NextDouble();
            }

        }

        // Randomly generate the starting weights of this layer (In this case between 1 and -1).
        public void GenerateStartWeights(double min, double max)
        {
            double weight;
            for (int i = 0; i < nodeNumber; i++)
            {

                for (int j = 0; j < nextLayerNodeNumber; j++)
                {
                    if (min < 0)
                    {
                        weight = rnd.NextDouble();
                        weight = weight * (max - min) + min;
                        this.weight[i, j] = weight;
                    }
                    else
                    {
                        weight = rnd.NextDouble();
                        weight = weight * (max + min) - min;
                        this.weight[i, j] = weight;
                    }

                }
            }
        }
    }
}

