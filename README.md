# Backpropagation

![Cover. Painting: The Course of Empire, Destruction by Thomas Cole, 1836](imgs/cover.jpg)

## Table of Contents

- [Backpropagation](#backpropagation)
  - [Table of Contents](#table-of-contents)
  - [Motivation](#motivation)
  - [Mathematical Foundation \& Theory](#mathematical-foundation--theory)
    - [Neural Network Basics](#neural-network-basics)
    - [Algorithm](#algorithm)
  - [Project Structure](#project-structure)
  - [Installation \& Usage](#installation--usage)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Usage](#usage)
  - [Data Bibliography](#data-bibliography)
  - [License](#license)

## Motivation

Imagine we're trying to fit a multi-layered model to data. In real life, we may start to nest multiple functions together. These functions need to be optimized. If we start with normal gradient descent, we can achieve this, but it will be very expensive computationally. Backpropagation is a numerical technique applied to gradient (or [other](https://github.com/intelligent-username/gradient-descent?tab=readme-ov-file#3adaptive) optimization methods) that makes this process practical.

## Mathematical Foundation & Theory

At its core, backpropagation takes reverse-mode differentiation and uses dynamic programming to make the calculation/storing process more efficient.

Automatic differentiation is among the simplest numerical methods in machine learning. Simply put when we have a 'nest' of functions $\hat{y} = f_n(f_{n-1}(...f_1(x))) + b$, and we want to find $\frac{\partial \hat{y}}{\partial x}$, we can use the chain rule, which gives:

$$
\frac{\partial y}{\partial x} = \frac{\partial f_n}{\partial f_{n-1}} \cdot \frac{\partial f_{n-1}}{\partial f_{n-2}} \cdots \frac{\partial f_1}{\partial x}
$$

In autodifferentiation, we read each partial derivative and multiply to get the final result. This can be done in two ways: **forward mode** and **reverse mode**.

Forward mode is when we read 'into' the function from the 'inside out', i.e. from the inputs expanding out to the outputs, $f_1$ out to $f_n$, calculating the values of the functions and derivatives as we go. Each intermediate derivative is computed alongside the function value.

So, we first find $f_1(x)$ and $\frac{\partial f_1}{\partial x}$, then use those to find $f_2(f_1(x))$ and $\frac{\partial f_2}{\partial f_1}$, and so forth.

In reverse mode, however, we calculate the values of the functions as we work 'into' the function, but then we propagate back out, finding the derivatives in the order $\frac{\partial{f_n}}{\partial x}$ ... $\frac{\partial{f_1}}{\partial x}$. The intermediate values from the forward pass are cached and later used to multiply the Jacobians in reverse. We do the same function evaluations first, cache results, and propagate sensitivities backward to compute derivatives.

In backpropagation, we use reverse mode since it's more efficient (there's almost always more neurons than outputs).

It's important to understand that a neural network is just terminology that is meant to help us intuit *how* the nested functions work. It works the same as any other optimization process, with training being a crucial part.

### Neural Network Basics

Imagine we have some data that follows the trend of the following equation:

$$\hat{y}(x,y) = f(g(x,y)) + h(g(x,y)) + e^{f(x)} + \sigma(x,y) + \sigma((g(x,y), f(x))) + b$$

Which can be visualized as a series of layers:

![Layered Neural Network](imgs/layers_example.png)

In real life, we're not just *given* equations, so we use neural networks to estimate what this function is going to be, based on just the features (inputs) and the labels (outputs).

Also, note that most neural networks are made up of more layers and conencted in more complex ways, this is just an illustrative example to see how to visualize the equations.

Some terminology:

- **Neural network**: the entire composite function.
- **Layer**: the set of neurons at the same 'depth' of the network.
- **Neuron**: a single unit within the neural network that applies a non-linear activation and a weighted sum. It's sometimes called a node. The process is to (1) take in some number of inputs, (2) apply weights and biases, and (3) pass the result to an activation function to get their result. This is always done in the same way.
- **Activation functions**: functions that determine the output of a neuron by applying a non-linear transformation to the input. Without them, we would just be composing a bunch of linear functions, which would be equivalent to a single linear function, thus regression to simple regression tasks instead of actually modelling complex curves.

![Activation Functions](imgs/a_f.png)

- **Perceptron**: Computes $y = f(wx+ b)$ for binary classification. Modern networks use differentiable activations in all layers, making classical perceptrons mostly historical.

Types of neural networks:

- **Dense**: in a dense neural network, every neuron in one layer is connected to every neuron in the next layer. All connections have a weight.
- **Sparse**: not every neuron is connected to every other neuron (in other words, some weights are zero).
&nbsp; - **Convolutional**: outputs are connected to a small patch of inputs.
&nbsp; - **Recurrent**: layers have connections looping back to previous layers, like an FSM.

Have a look at this beautiful diagram for a summary:

![Neural Network Summary](imgs/NN.png)

Now, we are ready to create the model. Every time a neuron receives its inputs, it weighs them and adds a bias. Then, it passes the result of this addition to an activation function, creating its output which is passed on to the next layer(s). When training a neural network, our goal is to find the best weights for making accurate predictions. We can turn to gradient descent, evolution algorithms, or other optimization methods to do this. In this project, I will focus on gradient descent.

When optimizing the weights between neurons through gradient descent, we can make many tweaks for the sake of efficiency and accuracy, but the biggest hurdle is the time it takes to compute the gradients. If we have a deep neural network with many layers and parameters, calculating the gradients is the limiting factor. This is why backpropagation is so important. However, at the same time, backpropagation isn't *strictly* necessary for making a neural network, though it is the most convenient way when using gradient descent.

### Algorithm

Backpropagation follows a simple mechanical process, based around gradient descent.

$$
w_{i+1} = w_i - \eta \frac{\partial L}{\partial w_i}
$$

In steps, this is:

1. **Initiate**: start with a neural network of weights and biases
2. **'Loss Term'**: Compute the forward pass to get network outputs, then compute the loss.
3. **Forward Pass**: compute the output of the network by evaluating each neuron from left to right, caching intermediate values.
4. **Backward Pass**: compute the derivative of the loss with respect to every parameter by applying the chain rule in reverse. Start from $\frac{\partial L}{\partial \hat{y}}$ and move backward through each layer, using cached values from the forward pass.
5. **Gradient Aggregation**: combine the local derivatives from each neuron to get the updated gradient.
6. **Parameter Update**: once all partial gradients are known, multiply by the learning rate and adjust the weights and biases.
7. **Iterate**: continue until convergence criteria are met (for example, predictions are very [similar](https://github.com/intelligent-username/Similarity-Metrics) to labels).

---

## Project Structure

```md
Backpropagation/
├── data/                   # The datasets used in the demos
├── imgs/                   # Just images for this writeup
├── models/                 # Saved models (pkl & h5)
├── nn/
│   ├── activations.py      # Activation functions & their derivatives
│   ├── gd.py               # Gradient Descent
│   ├── layer.py            # Layers, which are just matrices of weights that store past activations (store the information necessary for the math)
│   ├── loss.py             # Loss functions & their derivatives
│   └── network.py          # Actual Neural Network Structure, (basically just composition of the Layer class + function calls)
├── utils/
│   ├── loader.py           # Preprocessing the data
│   ├── metrics.py          # Return accuracy, specificity, and precision to evaluate the model.
│   ├── splitter.py         # Split data into training, validation, and test sets.
│   └── visualizer.py       # For visualizing the training and/or results. Has never and probably will never be actually implemented.
├── demo1.ipynb             # Demonstrating digit classification
├── demo2.ipynb             # Demonstrating student grade prediction
├── main.py                 # Small secondary explanation of how to use the library
```

## Installation & Usage

### Prerequisites

- Git
- Python 3.8.0 or higher
- Pip/Conda

### Setup

1. **Clone the repository from GitHub**:

    ```bash
    git clone https://github.com/intelligent-username/Backpropagation.git
    cd Backpropagation
    ```

2. **Install dependencies**:

    ```bash
    # Create and activate a virtual environment (optional)
    python -m venv venv               # Or through conda or something else :)
    source venv/bin/activate          # On Windows, use `venv\Scripts\activate`

    # Install the required packages
    pip install -r requirements.txt   # Or through another package manager if you prefer
    ```

### Usage

To use the neural network, import the `NeuralNetwork` class, define your layers and activations, and then train it on your data. Make sure to choose [appropriate loss functions](https://github.com/intelligent-username/Loss-Functions), activation functions, and the like.

Here's a quick example:

```python
  from nn.network import NeuralNetwork
  from nn.layer import Layer
  from nn.activations import tanh, tanh_derivative
  from nn.loss import MSE, MSE_derivative
  from nn.gd import fit

  # 1. Define your network architecture
  net = NeuralNetwork(loss=MSE, loss_derivative=MSE_derivative)
  net.add_layer(Layer(2, 3, tanh, tanh_derivative))
  net.add_layer(Layer(3, 1, tanh, tanh_derivative))

  # 2. Load your data (X_train, y_train, X_val, y_val, X_test, y_test)
  ...

  # 3. Then, train the network
  fit(net, X_train, y_train, X_val, y_val, X_test, y_test, max_epochs=1000, learning_rate=0.01)

  # 4. Make predictions
  # predictions = net.predict(X_test)
           # Evaluate accuracy, test with new data, etc. as needed
```

## Data Bibliography

So far, the demos make use of 2 datasets.

- [Student Grades](data/student_grades) database: Cortez, Paulo. "Student Performance." UCI Machine Learning Repository, 2008, [https://doi.org/10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).

- [Digits](data/digits) dataset is public domain. [https://www.kaggle.com/datasets/aquibiqbal/digits-09](https://www.kaggle.com/datasets/aquibiqbal/digits-09). Note that a lot of these have been compressed, etc. when pushing to GitHub, so you may want to re-install fresh. When using these, they will be resized and converted to grayscale.

## License

This project is distributed under the [MIT License](LICENSE).
