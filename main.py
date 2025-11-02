from nn.network import NeuralNetwork
from nn.layer import Layer
from nn.loss import cross_entropy, cross_entropy_derivative # Best for classification yo

from nn.activations import relu, relu_derivative, softmax, softmax_derivative_passthrough

from nn.gd import fit
from utils.loader import load_digits_dataset
from utils.splitter import data_split

import numpy as np

def main():
    print("Yo!!!! This file is going to do a small walkthrough on how to use the neural network library.")
    print("For actual real-life demos, go through the demos.ipynb file.")

    print("First, we have our loss functions and their derivatives. In loss.py, several are implemented. In the case of this project, I've only implemented the ones that would be required for the demos and some basic ones like MSE. Don't forget to import the loss functions that you need. The choice of a loss function is very important, for example the image classification task had about 30% accuracy with MSE but jumped to over 70% by just switching to cross-entropy.")

    print("Same goes with the activation functions and their derivatives.")

    print("Now, when creating the neural network, first make the layers (read the documentation to see what args are accepted, for example this is where activation functions are decided).")

    print("Then add the layers to the neural network in the order you need.")

    print("Finally, train the neural network using the gradient descent .fit() function. Tune the hyperparameters to get better results.")

    print("You can save the trained model by using the .save_model() method. It can later be reused. You can use the pickle library for Python-specific serialization, or the HDF5 format for interoperability with other frameworks, and even other languages.")
    print("If using the HDF5 format, make sure to convert the activation functions' names back to their callable forms (back to functions) by using the following mapping:")

    print("""
          activation_map = {
            "relu": relu_function,
            "sigmoid": sigmoid_function,
            "tanh": tanh_function,
            }

            for i, grp in enumerate(f["layers"]):
                act_name = grp.attrs["activation"]
                layer.activation = activation_map[act_name]
          
          """)

    #     activation_map = {
    #     "relu": relu_function,
    #     "sigmoid": sigmoid_function,
    #     "tanh": tanh_function,
    # }

    #     for i, grp in enumerate(f["layers"]):
    #         act_name = grp.attrs["activation"]
    #         layer.activation = activation_map[act_name]


if __name__ == "__main__":
    main()
