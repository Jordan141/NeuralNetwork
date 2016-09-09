using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace NeuralNetworkApp
{
    class Program
    {
        static void Main(string[] args)
        {
            string filePath = @"E:\Work\NeuralXOR.xml";

            int[] layerSizes = new int[3] { 2, 2, 1 };
            TransferFunction[] tFuncs = new TransferFunction[3] {TransferFunction.None,
                                                                   TransferFunction.Sigmoid,
                                                                   TransferFunction.Linear};
            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes,tFuncs);

            //Example XOR-Gate
            bpn.Name = "XOR-Gate";

            //Define the cases
            double[][] input, ouput;

            input = new double[4][]; ouput = new double[4][];
            for (int i = 0; i < 4; i++)
            {
                input[i] = new double[2]; ouput[i] = new double[1];
            }

            input[0][0] = 0.0; input[0][1] = 0.0; ouput[0][0] = 0.0; //false XOR false = false
            input[1][0] = 1.0; input[1][1] = 0.0; ouput[1][0] = 1.0; //true XOR false = true
            input[2][0] = 0.0; input[2][1] = 1.0; ouput[2][0] = 1.0; //false XOR true = true
            input[3][0] = 1.0; input[3][1] = 1.0; ouput[3][0] = 0.0; //true XOR true = false

            //Train the network
            double error = 0.0;
            int max_count = 1000, count = 0;

            do
            {
                //Prepare for training Epoch
                count++;
                error = 0.0;

                //Train
                for (int i = 0; i < 4; i++)
                    error += bpn.Train(ref input[i], ref ouput[i], 0.15, 0.10);

                //Show Progress
                if (count % 100 == 0)
                    Console.WriteLine("Epoch {0} completed with error {1:0.0000}", count, error);


            } while (error > 0.00001 && count <= max_count);

            //Display results!
            double[] networkOutput = new double[1];

            for(int i =0; i < 4; i++)
            {
                bpn.Run(ref input[i], out networkOutput);
                Console.WriteLine("Case {3}: {0:0.0} xor {1:0.0} = {2:0.0000}", input[i][0], input[i][1], networkOutput[0],i+1);
            }
            bpn.Save(filePath);



            Console.ReadLine();
        }
    }
}
