using System;

namespace LSTM_Spike
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin LSTM IO demo \n");

            Console.WriteLine("Creating an n=2 input, m=3 state LSTM cell");
            Console.WriteLine("Setting LSTM weights and biases to small arbitrary values \n");
            Console.WriteLine("Sending input = (1.0, 2.0) to LSTM \n");

            //input matrix
            float[][] xt = MatFromArray(new float[] { 1.0f, 2.0f }, 2, 1);

            //output matrix of previous outputs
            float[][] h_prev = MatFromArray(new float[] { 0.0f, 0.0f, 0.0f }, 3, 1);

            //cell state matrix
            float[][] c_prev = MatFromArray(new float[] { 0.0f, 0.0f, 0.0f }, 3, 1);

            //input gate weight matrix
            float[][] W = MatFromArray(new float[] { 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f }, 3, 2);

            //output gate weight matrices
            float[][] U = MatFromArray(new float[] { 0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f }, 3, 3);

            //bias matrices
            float[][] b = MatFromArray(new float[] { 0.16f, 0.17f, 0.18f }, 3, 1);

            /* legend
             * x(t): current input
             * h(t-1): previous output
             * c(t-1): previous cell state
             * h(t): current output
             * c(t): current cell state
             * f(t): forget gate
             * i(t): input gate
             * o(t): output gate
             */
                    
            /* these are copies of the basic structures previously listed
             * replicated for each of the gates and cell states */
            float[][] Wf = MatCopy(W); float[][] Wi = MatCopy(W);
            float[][] Wo = MatCopy(W); float[][] Wc = MatCopy(W);

            float[][] Uf = MatCopy(U); float[][] Ui = MatCopy(U);
            float[][] Uo = MatCopy(U); float[][] Uc = MatCopy(U);

            float[][] bf = MatCopy(b); float[][] bi = MatCopy(b);
            float[][] bo = MatCopy(b); float[][] bc = MatCopy(b);

            float[][] ht, ct;
            float[][][] result;

            result = ComputeOutputs(xt, h_prev, c_prev,
              Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc, bf, bi, bo, bc);

            ht = result[0];  // output
            ct = result[1];  // new cell state

            Console.WriteLine("Output is:");
            MatPrint(ht, 4, true);
            Console.WriteLine("New cell state is:");
            MatPrint(ct, 4, true);

            Console.WriteLine("=====");
            Console.WriteLine("\nSending input = (3.0, 4.0) to LSTM \n");

            h_prev = MatCopy(ht);
            c_prev = MatCopy(ct);
            xt = MatFromArray(new float[] { 3.0f, 4.0f }, 2, 1);

            result = ComputeOutputs(xt, h_prev, c_prev,
              Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc, bf, bi, bo, bc);

            ht = result[0];
            ct = result[1];

            Console.WriteLine("Output is:");
            MatPrint(ht, 4, true);
            Console.WriteLine("New cell state is:");
            MatPrint(ct, 4, true);

            Console.WriteLine("End LSTM demo ");

            Console.ReadLine();
        } // Main


        /* legend
         * x(t): current input
         * h(t-1): previous output
         * c(t-1): previous cell state
         * h(t): current output
         * c(t): current cell state
         * f(t): forget gate
         * i(t): input gate
         * o(t): output gate
         * U: output gate weights
         * W: input gate weights
         * b: bias weights
         */

        static float[][][] ComputeOutputs(float[][] xt, float[][] h_prev, float[][] c_prev,
          float[][] Wf, float[][] Wi, float[][] Wo, float[][] Wc,
          float[][] Uf, float[][] Ui, float[][] Uo, float[][] Uc,
          float[][] bf, float[][] bi, float[][] bo, float[][] bc)
        {
            float[][] ft = MatSig(MatSum(MatProd(Wf, xt), MatProd(Uf, h_prev), bf));
            float[][] it = MatSig(MatSum(MatProd(Wi, xt), MatProd(Ui, h_prev), bi));
            float[][] ot = MatSig(MatSum(MatProd(Wo, xt), MatProd(Uo, h_prev), bo));
            float[][] ct = MatSum(MatHada(ft, c_prev),
              MatHada(it, MatTanh(MatSum(MatProd(Wc, xt), MatProd(Uc, h_prev), bc))));
            float[][] ht = MatHada(ot, MatTanh(ct));

            float[][][] result = new float[2][][];
            result[0] = MatCopy(ht);
            result[1] = MatCopy(ct);
            return result;
        }

        // Matrix routines

        static float[][] MatCreate(int rows, int cols)
        {
            float[][] result = new float[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new float[cols];
            return result;
        }
        static float[][] MatFromArray(float[] arr, int rows, int cols)
        {
            if (rows * cols != arr.Length)
                throw new Exception("xxx");

            float[][] result = MatCreate(rows, cols);
            int k = 0;
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = arr[k++];
            return result;
        }

        static float[][] MatCopy(float[][] m)
        {
            int rows = m.Length; int cols = m[0].Length;
            float[][] result = MatCreate(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = m[i][j];
            return result;
        }
        static float[][] MatProd(float[][] a, float[][] b)
        {
            int aRows = a.Length; int aCols = a[0].Length;
            int bRows = b.Length; int bCols = b[0].Length;
            if (aCols != bRows)
                throw new Exception("xxx");
            float[][] result = MatCreate(aRows, bCols);
            for (int i = 0; i < aRows; ++i) // each row of a
                for (int j = 0; j < bCols; ++j) // each col of b
                    for (int k = 0; k < aCols; ++k) // could use k < bRows
                        result[i][j] += a[i][k] * b[k][j];
            return result;
        }

        // element-wise functions

        static float[][] MatSig(float[][] m)
        {
            // element-wise sigmoid
            int rows = m.Length; int cols = m[0].Length;

            float[][] result = MatCreate(rows, cols);
            for (int i = 0; i < rows; ++i) // each row
                for (int j = 0; j < cols; ++j) // each col
                    result[i][j] = Sigmoid(m[i][j]);
            return result;
        }

        static float[][] MatTanh(float[][] m)
        {
            // element-wise tanh
            int rows = m.Length; int cols = m[0].Length;

            float[][] result = MatCreate(rows, cols);
            for (int i = 0; i < rows; ++i) // each row
                for (int j = 0; j < cols; ++j) // each col
                    result[i][j] = Tanh(m[i][j]);
            return result;
        }

        static float Sigmoid(float x)
        {
            if (x < -10.0) return 0.0f;
            else if (x > 10.0) return 1.0f;
            return (float)(1.0 / (1.0 + Math.Exp(-x)));
        }

        static float Tanh(float x)
        {
            if (x < -10.0) return -1.0f;
            else if (x > 10.0) return 1.0f;
            return (float)(Math.Tanh(x));
        }
        static float[][] MatHada(float[][] a, float[][] b)
        {
            // Hadamard element-wise multiplication
            // assumes a, b have same shape
            int rows = a.Length; int cols = a[0].Length;

            float[][] result = MatCreate(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = a[i][j] * b[i][j];
            return result;
        }

        static float[][] MatSum(float[][] a, float[][] b)
        {
            int rows = a.Length; int cols = a[0].Length;

            float[][] result = MatCreate(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = a[i][j] + b[i][j];
            return result;
        }

        static float[][] MatSum(float[][] a, float[][] b, float[][] c)
        {
            int rows = a.Length; int cols = a[0].Length;

            float[][] result = MatCreate(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = a[i][j] + b[i][j] + c[i][j];
            return result;
        }
        static void MatPrint(float[][] Mat, int dec, bool nl)
        {
            for (int i = 0; i < Mat.Length; ++i)
            {
                for (int j = 0; j < Mat[0].Length; ++j)
                {
                    Console.Write(Mat[i][j].ToString("F" + dec) + " ");
                }
                Console.WriteLine("");
            }
            if (nl == true) Console.WriteLine("");
        }

    } // Program
}
