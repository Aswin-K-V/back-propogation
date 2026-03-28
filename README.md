# Back Propagation From Scratch

Small educational Python project that implements:

- a scalar autograd engine
- a tiny multilayer perceptron (MLP)
- an XOR training demo with a decision-boundary visualization

The repository is useful if you want to understand how backpropagation works without relying on a deep learning framework.

This project is also referenced from Andrej Karpathy's `micrograd` repository and follows the same educational idea of building autograd and a small neural network from first principles.

## Project Structure

- `engine.py` - defines the `Value` class used for scalar computation graphs and automatic differentiation
- `neural_network.py` - builds `Module`, `Neuron`, `Layer`, and `MLP` on top of the autograd engine
- `demo_xor_visualization.py` - trains the network on XOR and visualizes the learned boundary over time

## Requirements

Core model code only uses the Python standard library.

To run the visualization demo, install:

- `numpy`
- `matplotlib`

Example:

```bash
pip install numpy matplotlib
```

## Run The Demo

```bash
python demo_xor_visualization.py
```

What the demo does:

1. Builds a `2 -> 4 -> 4 -> 1` MLP.
2. Trains it on the XOR dataset using squared error loss.
3. Updates parameters with manual gradient descent.
4. Animates how the decision boundary changes during training.

If Matplotlib is using a non-interactive backend, the script will still train the model and print final predictions, but it will skip the live animation window.

## Example Output

```text
Final XOR predictions:
  input=[0.0, 0.0] predicted=-0.0000 target=0.0
  input=[0.0, 1.0] predicted=1.0000 target=1.0
  input=[1.0, 0.0] predicted=1.0000 target=1.0
  input=[1.0, 1.0] predicted=-0.0000 target=0.0
Final loss: 0.00000000
```

## How It Works

`engine.py` tracks a computation graph for scalar values. Each operation stores:

- the output value
- references to parent nodes
- a local `_backward()` function

Calling `backward()` performs a topological traversal of the graph and propagates gradients in reverse order.

`neural_network.py` uses these scalar `Value` objects as trainable weights and biases. Because every neuron output is built from `Value` operations, the full network becomes differentiable automatically.

