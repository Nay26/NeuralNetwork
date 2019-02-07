using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            // Initailise The Neural Network.
            NeuralNetwork NumberGuesser = new NeuralNetwork(784,16,1,10);
            NumberGuesser.InitialiseNetwork();
            NumberGuesser.InitialiseWeights();

            // User input to decide what to do.
            string input;
            do
            {
                Console.WriteLine("t to Train, r for Run, p for Printed Run, q to Quit:");
                input = Console.ReadLine();
                if (!(input.Equals("q")))
                {
                    NumberGuesser = Run(NumberGuesser, input);
                }
            } while (!(input.Equals("q")));
                 
        }

        // The Neural Network Run fucntion.
        public static NeuralNetwork Run(NeuralNetwork nn, string input)
        {
            // Get the training data labels and images from the files.
            FileStream ifsLabels =
            new FileStream(@".\train-labels.idx1-ubyte",
            FileMode.Open); // test labels
            FileStream ifsImages =
             new FileStream(@".\train-images.idx3-ubyte",
             FileMode.Open); // test images

            BinaryReader brLabels =
             new BinaryReader(ifsLabels);
            BinaryReader brImages =
             new BinaryReader(ifsImages);

            int magic1 = brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();

            int magic2 = brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            // Creating a byte array to store image data.
            byte[][] pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new byte[28];


            double total = 0;

            // Loop for each image in the training set.
            for (int di = 0; di < 10000; ++di)
            {
                // Load the image into the byte array.
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i][j] = b;
                    }
                }

                // Get the label (What Image should be).
                byte lbl = brLabels.ReadByte();

                //The first half (5000) images of the training data are reserved fro training
                if (di < 5000) { 
                    if (input.Equals("t"))
                    {
                        // Method to get the input values for the input layer of the network
                        nn.InitialiseInputs(pixels);

                        //Forward Propogate through the network using the input values and the weights
                        nn.ForwardPropogate();

                        //If the Network returns the correct value (same as label) increment the total.
                        if (nn.IsOutputCorrect(lbl) == true)
                        {
                            total++;
                        }

                        //Since this is the training step, perform the backpropogation algorithm on the network
                        nn.BackPropogate(lbl);
                    }
                }

                //The second half (5000) images of the training data are reserved for seeing how well the neural netwrok performs
                if (di > 5000)
                {
                    if (!(input.Equals("t")))
                    {
                        // Method to get the input values for the input layer of the network
                        nn.InitialiseInputs(pixels);

                        //Forward Propogate through the network using the input values and the weights
                        nn.ForwardPropogate();

                        // If the user selected a printed run then show the process step by step, printing a representation of the handwriting
                        if (input.Equals("p"))
                        {
                            nn.PrintNetwork();
                            DigitImage digit = new DigitImage(pixels, lbl);
                            Console.WriteLine(digit.ToString());
                            Console.WriteLine("I think this was: " + nn.OutputValue());
                            Console.ReadLine();
                        }

                        //If the Network returns the correct value (same as label) increment the total.
                        if (nn.IsOutputCorrect(lbl) == true)
                        {
                            total++;
                        }
                    }        
                }

            }

            //Once run has finished show the percentage of images the network guessed correctly
            Console.WriteLine((total / 5000) * 100 + "%");

            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

            Console.WriteLine("\nEnd of Running\n");
            return nn;
        }                           

    }
}
