using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    public class DigitImage
    {
        public byte[][] pixels;
        public byte label;

        public DigitImage(byte[][] pixels,
          byte label)
        {
            this.pixels = new byte[28][];
            for (int i = 0; i < this.pixels.Length; ++i)
                this.pixels[i] = new byte[28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixels[i][j] = pixels[i][j];

            this.label = label;
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (this.pixels[i][j] < 25)
                        s += " "; // white
                    else if (this.pixels[i][j] < 50)
                        s += "."; // white
                    else if (this.pixels[i][j] < 75)
                        s += ":"; // white
                    else if (this.pixels[i][j] < 100)
                        s += "-"; // white
                    else if (this.pixels[i][j] < 125)
                        s += "="; // white
                    else if (this.pixels[i][j] < 150)
                        s += "+"; // white
                    else if (this.pixels[i][j] < 175)
                        s += "*"; // white
                    else if (this.pixels[i][j] < 200)
                        s += "#"; // white
                    else if (this.pixels[i][j] < 225)
                        s += "%"; // white
                    else if (this.pixels[i][j] < 255)
                        s += "@"; // white
                }
                s += "\n";
            }
            s += this.label.ToString();
            return s;
        } // ToString
    }

}

