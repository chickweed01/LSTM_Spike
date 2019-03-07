# LSTM via UML
> An exploration of Long Short-term Memory Cells using a UML-based approach

## Introduction
In my quest to have a better understanding if how Long Short-Term Memory (LSTM) cells function I decided to model a cell using UML and implement it in C#. (I'll assume that you are familar with the use of LSTM cells in recurrent neural networks, so I won't cover it here. This [blog](https://bit.ly/1iaBaLH) provides a great explanation.)  The question then became how to validate that my implementation was accurate. Fortunately, I came across an [article](https://bit.ly/2NMdn5n) in MSDN Magazine by James McCaffrey of Microsoft Research. In this article Dr. McCaffrey describes an implemention of a LSTM cell written in C#. With Dr McCaffrey's program in hand to be used as a benchmark I set out to create a model of an LSTM cell and its components expressed as UML classes.

## The Model

The basic function of an LSTM cell is to determine how much information about the previous cell state to persist in the cell's current data. That determination is made using a series of "gates". A diagram of a typical cell is shown below:

![Image](/LSTM_UML/images/LSTMCell.jpg?raw=true "Title")


The first gate in the series is called the "Forget Gate".

![Image](/LSTM_UML/images/ForgetGateSummations.jpg?raw=true "Title")

