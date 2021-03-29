 핸즈온 컴퓨터 비전 & 텐서플로우 2 (2019) 필사

[챕터 1 주피터 노트북](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/blob/master/Chapter01/ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)
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

### Layering Neurons Together
```python
class FullyConnectedLayer(object):
    """A simple fully-connected NN layer.
    Args:
        num_inputs (int): The input vector size / number of input values.
        layer_size (int): The output vector size / number of neurons in the layer.
        activation_function (callable): The activation function for this layer.
    Attributes:
        W (ndarray): The weight values for each input.
        b (ndarray): The bias value, added to the weighted sum.
        size (int): The layer size / number of neurons.
        activation_function (callable): The activation function computing the neuron's output.
        x (ndarray): The last provided input vector, stored for backpropagation.
        y (ndarray): The corresponding output, also stored for backpropagation.
        derivated_activation_function (callable): The corresponding derivated function for backpropagation.
        dL_dW (ndarray): The derivative of the loss, with respect to the weights W.
        dL_db (ndarray): The derivative of the loss, with respect to the bias b.
    """
    def __init__(self, num_inputs, layer_size, activation_function, derivated_activation_function=None):
        super().__init__()

        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size

        self.activation_function = activation_function
        self.derivated_activation_function = derivated_activation_function
        self.x, self.y = None, None
        self.dL_dW, self.dL_db = None, None

    def forward(self, x):
        """
        Forward the input vector through the layer, returning its activation vector.
        Args:
            x (ndarray): The input vector, of shape `(batch_size, num_inputs)`
        Returns:
            activation (ndarray): The activation value, of shape `(batch_size, layer_size)`.
        """
        z = np.dot(x, self.W) + self.b
        self.y = self.activation_function(z)
        self.x = x # we store the input and output values for back-propagation

        return self.y

    def backward(self, dL_dy):
        """
        Back-propagate the loss, computing all the derivatives, storing those w.r.t. the layer parameters,
        and returning the loss w.r.t. its inputs for further propagation.
        Args:
            dL_dy (ndarray): The loss derivative w.r.t. the layer's output (dL/dy = l'_{k+1}).
        Returns:
            dL_dx (ndarray): The loss derivative w.r.t. the layer's input (dL/dx).
        """

        dy_dz = self.derivated_activation_function(self.y) # = f'
        dL_dz = (dL_dy * dy_dz) # dL/dz = dL/dy * dy/dz = l '_{k+1} * f'
        dz_dw = self.x.T    
        dz_dx = self.W.T    
        dz_db = np.ones(dL_dy.shape[0]) # dz/db = d(W.x + b) /db = 0 + db/db = "ones" -vector

        # Computing the derivtives with respect to the layer's parameters, and storing them for opt. optimization:
        self.dL_dW = np.dot(dz_dw, dL_dz)
        self.dL_db = np.dot(dz_db, dL_dz)

        # Computing the derivative with respect to the input, to be passed to the previous layers (their 'dL_dy'):
        dL_dx = np.dot(dL_dz, dz_dx)
        return dL_dx

    def optimize(self, epsilon):
        """
        Optimize the layer's parameters, using the stored derivative values.
        Args:
            epsilon (float): The learning rate.
        """
        self.W -= epsilon * self.dL_dW
        self.b -= epsilon * self.dL_db



input_size = 2
num_neurons = 3
relu_function = lambda y: np.maximum(y, 0)

layer = FullyConnectedLayer(num_inputs=input_size, layer_size=num_neurons, activation_function=relu_function)

x1 = np.random.uniform(-1, 1, 2).reshape(1, 2)
print("Input vector #1: {}".format(x1))

x2 = np.random.uniform(-1, 1, 2).reshape(1, 2)
print("Input vecotr #2: {}".format(x2))

y1 = layer.forward(x1)
print("Layer's output value given `x1`: {}".format(y1))

y2 = layer.forward(x2)
print("Layer's output value given `x2`: {}".format(y2))

x12 = np.concatenate ((x1, x2))  # stack of input vector, of shape `(2,2)`
y12 = layer.forward(x12)
print("Layer's output value given `[x1, x2] : \n{}".format(y12))

```
Result
```
Input vector #1: [[ 0.60439396 -0.85089871]]
Input vecotr #2: [[0.97377387 0.54448954]]
Layer's output value given `x1`: [[0.         0.         0.32338208]]
Layer's output value given `x2`: [[0.39931638 0.         0.        ]]
Layer's output value given `[x1, x2] : 
[[0.         0.         0.32338208]
 [0.39931638 0.         0.        ]]
```
