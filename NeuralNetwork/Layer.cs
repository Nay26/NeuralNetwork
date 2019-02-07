using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Layer
    {

        public double[] nodeBias;
        public double[] nodeOutput;
        public double[,] weight;
        public double[,] weightErrorDerivative;
        public double[] activationDerivative;
        public double[] ENArray;
        public int nodeNumber;
        public int nextLayerNodeNumber;
        Random rnd;

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

