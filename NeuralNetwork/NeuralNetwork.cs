using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    //The Neural Network Object
    class NeuralNetwork
    {
        // Contains a list of Layers.
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

        // On initialisation create the layer list of layers.
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

        // Setting the input layer node values that correspond to the darkness of the pixels of the training image
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

        // Generate the start weights for each layer in the layer list
        public void InitialiseWeights()
        {
            foreach (Layer layer in layers)
            {
                layer.GenerateStartWeights(minStartWeight, maxStartWeight);
            }
        }

        // The simple forward propogation algorithm that runs the input values through the network to return an output of what the network thinks the image is.
        public void ForwardPropogate()
        {
            
            for (int k = 0; k < layers.Count; k++) // For each layer
            {
                for (int i = 0; i < layers[k].nodeNumber; i++) // For each node in that layer
                {
                    if (k > 0) // If Not Input Layer ie: one step done, run the activation function and add the bias before calculating the next layers outputs.
                    {
                        layers[k].nodeOutput[i] = SigmoidFunction((layers[k].nodeOutput[i]) + layers[k].nodeBias[i]);
                    }
                    for (int j = 0; j < layers[k].nextLayerNodeNumber; j++) // For each node in the next layer calculate it's output as the sum of all the weights going into it * thier input values.
                    {
                        layers[k + 1].nodeOutput[j] += layers[k].weight[i, j] * layers[k].nodeOutput[i];
                    }
                }
            }
        }

        // Sigmoid Activation Function is used in this network, derivative function is needed for backpropogation step
        public double SigmoidFunction(double x)
        {
            return 1 / (1 + (Math.Pow(Math.E, -x)));
        }

        public double DerivSigmoid(double x)
        {
            return x * (1 - x);
        }

        // Backpropogation Algorithm (Only working for one hidden Layer)
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

        // Printing the network for test purposes.
        internal void PrintNetwork()
        {
            for (int i = 0; i < layers[layers.Count - 1].nodeNumber; i++)
            {
                Console.WriteLine("Node value " + i + " " + layers[layers.Count - 1].nodeOutput[i]);
            }
        }

        // Is the output of the network equal to the actual image?
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

        // Calculate what the network thinks the output value is.
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

        
    }
}

