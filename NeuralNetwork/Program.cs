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
            NeuralNetwork NumberGuesser = new NeuralNetwork(784,16,1,10);
            NumberGuesser.InitialiseNetwork();
            NumberGuesser.InitialiseWeights();          
            string input = "r";
            NumberGuesser = Run(NumberGuesser, input);
            do
            {
                Console.WriteLine("T to Train, R for Run, P for printed Run, G to move to Generation :");
                input = Console.ReadLine();         
                if ((!(input.Equals("g"))))
                {
                    NumberGuesser = Run(NumberGuesser, input);
                }      
            } while (!(input.Equals("g")));       
        }

        public static NeuralNetwork Run(NeuralNetwork nn, string input)
        {
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

            byte[][] pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new byte[28];


            double total = 0;
            // each test image
            for (int di = 0; di < 10000; ++di)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i][j] = b;
                    }
                }
                byte lbl = brLabels.ReadByte();
                if (di < 5000) { 
                    if (input.Equals("t"))
                    {
                        nn.InitialiseInputs(pixels);
                        nn.ForwardPropogate();
                        if (nn.IsOutputCorrect(lbl) == true)
                        {
                            total++;
                        }
                        nn.BackPropogate(lbl);
                    }
                }
                if (di > 5000)
                {
                    if (!(input.Equals("t")))
                    {
                        nn.InitialiseInputs(pixels);
                        nn.ForwardPropogate();
                        if (input.Equals("p"))
                        {
                            nn.PrintNetwork();
                            DigitImage digit = new DigitImage(pixels, lbl);
                            Console.WriteLine(digit.ToString());
                            Console.WriteLine("I think this was: " + nn.OutputValue());
                            Console.ReadLine();
                        }
                        if (nn.IsOutputCorrect(lbl) == true)
                        {
                            total++;
                        }
                    }        
                }

            }

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
