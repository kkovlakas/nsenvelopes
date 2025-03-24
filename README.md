# nsenvelopes

## Introduction

(Magneto-)thermal simulations of neutron stars rely on the Tb-Ts relation,
that is the relation between the bottom and surface  temperature of the envelope
of the neutron star. While published parametric fits exist, new physics and
additional parameters require the construction of new relations, based on
envelope simulations. The `nsenvelopes` package aims at providing a framework
for training simple neural networks that approximate the Tb-Ts relation.

## Installation

Installation is required **only** if you wish to train your own neural networks to approximate the Tb-Ts relation, or any other relation of similar structure (e.g., initial-final value of a quantity that evolves in space, time, or another variable.)

Simply download or clone the source, switch to your favourite environment, and run

`pip install .`

from the top level of the repository.

## Data

In the folder `data/20241029` you will find the trained models as of 2024/10/29 for the Fe and H envelopes, in two formats: exported `tensorflow` model (`*_TF.zip`) or text files containing the weights and biases of the networks (`*.txt.zip`):
* `Fe_TF.zip`: tensorflow model for the Fe envelope
* `H_TF.zip`: same as above for the H envelope
* `Fe.txt.zip`: weights/biases to be loaded manually for the Fe enevelope
* `H.txt.zip`: same as above for the H envelope

See **Examples** for instructions on reading and using these files.

## Examples (no installation required)

These are examples of using models trained using `nsenvelopes`, but do not require the package to be loaded. In fact, `tensorflow` can be used to load them (Example 1), or alternatively, a custom function to load the model from a text file, having no dependency other than `numpy`.

### Example 1: Reading a trained `tensorflow` model

The trained models are provided as exported `tensorflow` models, and therefore
can be used directly by all platforms and programming languages supported by
`tensorflow` (e.g., Python, C++, Javascript). Here we show how the model can be
loaded from its folder (e.g., after unzipping the models in `data/20241029`), and applied for a given set of input parameters:

```python
import tensorflow as tf
loaded_model = tf.keras.models.load_model("models/Fe")
X = [[8.0, 7.5, 14, 0.0]]       # input (log_rhob and log_Tb, logB and Theta_B)
logT_s = model.predict(X)       # output (log_Ts)
```

where directly applying it to multiple input values
(two-dimensional array) allows to exploit the multi-processing
capabilities of neural networks:

```python
logT_s_grid = model.predict(input_2d)
```

### Example 2: Reading weights from a text file

For compatibility with any programming language, we also provide text files (`data/20241029/*.txt.zip`) that list all weights and biases. The first two lines
holds the number of inputs $m$ (here, 4) and the number of units in the hidden layer $n$. Then, the weights between the input and
hidden layer follow: $n$ lines, each with $m$ space-separated numbers. Consequntly, $n$ lines representing the biases in the hidden layers, and another $n$ lines for the weights between the hidden and the output layer. The last line holds the bias of the output unit.

For example:

```plain
4
2048
0.00080214 -0.00008899 0.00195929 -0.00178556
...
1.18483090
```

For example, the following code shows how to read the weights/biases in Python, but can easily be adapted for different languages:

```python
import numpy as np

def parse_numbers(line, n_expected=1, is_float=True):
    converter = float if is_float else int
    numbers = [converter(word) for word in line.strip().split()]
    assert len(numbers) == n_expected
    return numbers[0] if n_expected == 1 else numbers

def read_weights(filepath):
    with open(filepath, "rt", encoding="utf-8") as f:
        lines = iter(f)
        n_input = parse_numbers(next(lines), 1, is_float=False)
        n_units = parse_numbers(next(lines), 1, is_float=False)
        w_hidden = [parse_numbers(next(lines), n_input) for _ in range(n_units)]
        b_hidden = [parse_numbers(next(lines, 1)) for _ in range(n_units)]
        w_output = [parse_numbers(next(lines, 1)) for _ in range(n_units)]
        b_output = parse_numbers(next(lines), 1)
        assert not len(list(lines)) # no remaining lines
        return [np.array(arr) for arr in [w_hidden, b_hidden, w_output, b_output]]

w_hid, b_hid, w_out, b_out = read_weights("Fe.txt")
```

Then, the feedforward propagation step is simply

```python
np.dot(w_out, 1 / (1 + np.exp(-(np.dot(w_hid, X) + b_hid)))) + b_out
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Authors

* Konstantinos Kovlakas (kkovla@gmail.com)
* Davide De Grandis
