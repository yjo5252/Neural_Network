 핸즈온 컴퓨터 비전 & 텐서플로우 2 (2019) 필사

## Chapter 1

### 1.1 Building and Training a Neural Network from Scratch 
* Implement a simple  neural network, from the <i> modeling of an artificial neuron</i> to a <i>multi-layered system </i> which can be trained to classify images of hand-written digits.
* neuron.py: model of an <i>artificial neuron</i> able to forward information.
* fully_connected_layer.py : implementation of a funcional <i> layer</i> grouping several neurons, with methods to optimize its parameters.
* simple_network.py: class wrapping everything togehter into a modular <i> neural network </i> model which can be traiend and used for various tasks. 

### At the Beginning : the Neuron
```Python
import numpy as np     # numpy used to do vector and matrix computations
np.random.seed(42) 

class Neuron(object):
  """
    A simple artificial neuron, processing an input vector and returning a corresponding activation.
    Args:
        num_inputs (int): The input vector size / number of input values.
        activation_function (callable): The activation function defining this neuron.
    Attributes:
        W (ndarray): The weight values for each input.
        b (float): The bias value, added to the weighted sum.
        activation_function (callable): The activation function computing the neuron's output.
    """

  def __init__(self, num_inputs, activation_function):
      super().__init__()

      # Randomly initializing the weight vector and the bias value (e.g., using a simplistic 
      # uniform distribution between -1 and 1):

      self.W = np.random.uniform(size=num_inputs, low=-1., high=1.)
      self.b = np.random.uniform(size=1, low=-1., high=1.)
      self.activation_function = activation_function


  def forward(self, x):
      """
        Forward the input signal through the neuron, returning its activation value.
        Args:
            x (ndarray): The input vector, of shape `(1, num_inputs)`
        Returns:
            activation (ndarray): The activation value, of shape `(1, layer_size)`.
      """

      z = np.dot(x, self.W) + self.b
      return self.activation_function(z)


# instantiate our neuron
# create a neuron (perceptron)

# Pereptron input size:
input_size = 3

# Step function (returns 0 if y <= 0, or 1 if y > 0)
step_function = lambda y: 0 if y <= 0 else 1 

# Instantiating the perceptron
perceptron = Neuron(num_inputs=input_size, activation_function=step_function)
print("Perceptron's random weights = {}, and random bias = {}".format(perceptron.W, perceptron.b))

# randomly generate a random input vector of 3 values (a column-vector of shape=(1,3))
x = np.random.rand(input_size).reshape(1, input_size)
print("Input vector: {}".format(x))

# feed our perceptron with this input and display the corresponding activation.
y = perceptron.forward(x)
print("Perceptron's output value given `x`: {}".format(y))

```
<i>result</i>
```
Perceptron's random weights = [-0.25091976  0.90142861  0.46398788], and random bias = [0.19731697]
Input vector: [[0.15601864 0.15599452 0.05808361]]
Perceptron's output value given `x`: 1
```
