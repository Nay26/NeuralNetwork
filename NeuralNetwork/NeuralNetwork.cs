using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class NeuralNetwork
    {
        List<Layer> layers;

        double maxStartWeight;
        double minStartWeight;

        public int inputLayerSize;
        public double[] inputLayerInputs;

        public int numberOfHiddenLayers;
        public int hiddenLayerSize;

        public int outputLayerSize;

        public double[] deltaArray;

        public NeuralNetwork(int inputs, int hiddens, int hiddennum, int outputsize)
        {

            layers = new List<Layer>();
            maxStartWeight = 1;
            minStartWeight = -1;
            inputLayerSize = inputs;
            numberOfHiddenLayers = hiddennum;
            hiddenLayerSize = hiddens;
            outputLayerSize = outputsize;

        }

        public void InitialiseNetwork()
        {
            Layer inputLayer = new Layer(inputLayerSize, hiddenLayerSize);
            layers.Add(inputLayer);

            for (int i = 0; i < numberOfHiddenLayers - 1; i++)
            {
                Layer hiddenLayer = new Layer(hiddenLayerSize, hiddenLayerSize);
                layers.Add(hiddenLayer);
            }

            Layer hiddenLayerEnd = new Layer(hiddenLayerSize, outputLayerSize);
            layers.Add(hiddenLayerEnd);

            Layer outputLayer = new Layer(outputLayerSize, 0);
            layers.Add(outputLayer);
        }

        public void InitialiseInputs(byte[][] pixels)
        {
            inputLayerInputs = new double[inputLayerSize];
            double tempInput;
            int count = 0;
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    if (pixels[i][j] == 0)
                    {
                        inputLayerInputs[count] = 0;

                    }
                    else
                    {
                        tempInput = (pixels[i][j]);
                        inputLayerInputs[count] = tempInput / 255;

                    }

                    count++;
                }
            }
            layers[0].nodeOutput = inputLayerInputs;
        }

        public void InitialiseWeights()
        {
            foreach (Layer layer in layers)
            {
                layer.GenerateStartWeights(minStartWeight, maxStartWeight);
            }
        }

        public void ForwardPropogate()
        {
            for (int k = 0; k < layers.Count; k++)
            {
                for (int i = 0; i < layers[k].nodeNumber; i++)
                {
                    if (k > 0)
                    {
                        layers[k].nodeOutput[i] = SigmoidFunction((layers[k].nodeOutput[i]) + layers[k].nodeBias[i]);
                    }


                    for (int j = 0; j < layers[k].nextLayerNodeNumber; j++)
                    {
                        layers[k + 1].nodeOutput[j] += layers[k].weight[i, j] * layers[k].nodeOutput[i];
                    }
                }
            }
        }

        public double SigmoidFunction(double x)
        {
            return 1 / (1 + (Math.Pow(Math.E, -x)));
        }

        public double DerivSigmoid(double x)
        {
            return x * (1 - x);
        }

        public void BackPropogate(int actual)
        {
            CalculateActivationDerivatives();
            CalculateOutputErrorDerivative(actual);

            for (int l = layers.Count - 1; l > 0; l--)
            { 

                for (int bn = 0; bn < layers[l].nodeNumber; bn++)
                {
                    for (int fn = 0; fn < layers[l-1].nodeNumber; fn++)
                    {
                        layers[l-1].weightErrorDerivative[fn, bn] = ExtraBit(l,fn,bn) * layers[l].activationDerivative[bn] * layers[l-1].nodeOutput[fn];
                    }
                }

            }         

            AdjustWeights();
        }

        public double ExtraBit(int layer, int fn, int bn)
        {
            if (layer == layers.Count - 1)
            {
                return deltaArray[bn];
            }
            else if (layer == layers.Count - 2)
            {
                double EN = 0;
                if (!(fn == 0))
                {
                    EN = layers[layer].ENArray[bn];
                }
                else
                {
                    for (int x = 0; x < outputLayerSize; x++)
                    {
                        EN = EN + deltaArray[x] * layers[layer + 1].activationDerivative[x] * layers[layer].weight[bn, x];
                    }
                    layers[layer].ENArray[bn] = EN;
                }               
                return EN;
            }
            else if (layer == layers.Count - 3)
            {
                double EN = 0;
                if (!(fn == 0))
                {
                    EN = layers[layer].ENArray[bn];
                }
                else
                {
                    for (int x = 0; x < bn; x++)
                    {
                        EN = EN + layers[layer+1].ENArray[x] * layers[layer + 1].activationDerivative[x] * layers[layer].weight[bn, x];
                    }
                    layers[layer].ENArray[bn] = EN;
                }
                return EN;
            }
            else
            {
                return 1;
            }
        }

        public void AdjustWeights()
        {
            double learningRate = 0.05;
            for (int l = 0; l < layers.Count; l++)
            {
                for (int i = 0; i < layers[l].nodeNumber; i++)
                {
                    for (int j = 0; j < layers[l].nextLayerNodeNumber; j++)
                    {
                        layers[l].weight[i, j] += layers[l].weightErrorDerivative[i, j] * learningRate;
                    }
                }
            }

        }

        internal void PrintNetwork()
        {
            for (int i = 0; i < layers[layers.Count - 1].nodeNumber; i++)
            {
                Console.WriteLine("Node value " + i + " " + layers[layers.Count - 1].nodeOutput[i]);
            }
        }

        public Boolean IsOutputCorrect(byte actual)
        {
            if (OutputValue() == actual)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public void GetTotalError()
        {

        }

        public int OutputValue()
        {
            double max = -100;
            int maxNodeNum = -1;
            for (int i = 0; i < layers[layers.Count - 1].nodeNumber; i++)
            {
                if (layers[layers.Count - 1].nodeOutput[i] > max)
                {
                    max = layers[layers.Count - 1].nodeOutput[i];
                    maxNodeNum = i;
                }

            }
            return maxNodeNum;
        }

        public void CalculateActivationDerivatives()
        {
            for (int l = 0; l < layers.Count; l++)
            {
                for (int n = 0; n < layers[l].nodeNumber; n++)
                {
                    layers[l].activationDerivative[n] = DerivSigmoid(layers[l].nodeOutput[n]);

                }
            }
        }

        public void CalculateOutputErrorDerivative(int actual)
        {
            deltaArray = new double[outputLayerSize];

            for (int i = 0; i < deltaArray.Length; i++)
            {
                if (actual == i)
                {
                    deltaArray[i] = (1 - layers[layers.Count - 1].nodeOutput[i]);
                }
                else
                {
                    deltaArray[i] = (0 - layers[layers.Count - 1].nodeOutput[i]);
                }
            }
        }
    }
}

