"""Tuning and training of single hidden layer feedforward neural networks."""


import numpy as np

import tensorflow as tf
from tensorflow.keras.regularizers import L1, L2, L1L2
import keras_tuner


DEFAULT_LAYER_WIDTHS = [256, 512, 1024, 2048, 4096]
DEFAULT_REGULARIZATION_FACTORS = [1.0e-4, 1.0e-5, 1.0e-6]
DEFAULT_LEARNING_RATES = [1.0e-3, 1.0e-2, 1.0e-1]


VALID_INITIALIZERS = ["zeros", "ones", "glorot_uniform", "glorot_normal",
                      "he_uniform", "he_normal", "uniform", "random_normal",
                      "lecun_uniform", "lecun_normal", "truncated_normal"]


VALID_ACTIVATIONS = ["relu", "sigmoid", "tanh", "softmax", "softplus",
                     "softsign", "exponential", "linear", "selu", "elu",
                     "gelu", "swish", "hard_sigmoid"]


VALID_REGULARIZERS = ["L1L2", "L1", "L2", None]


class ModelArchitect:
    """Class for tuning and training SLFN models."""
    def __init__(self,
                 subsets,
                 features,
                 targets,
                 separate_holdout=True,
                 layer_widths=None,
                 learning_rates=None,
                 regularization_factors=None,
                 regularizer="L1L2",
                 initializer="he_normal",
                 activation="sigmoid"):
        self.subsets = subsets
        self.features = features
        self.targets = targets
        self.separate_holdout = separate_holdout
        self.layer_widths = layer_widths
        self.learning_rates = learning_rates
        self.regularization_factors = regularization_factors
        self.regularizer = regularizer
        self.initializer = initializer
        self.activation = activation

        self.tuner = None
        self.best_model = None
        self.selected_model = None

        self._validity_check()

    def _validity_check(self):
        if self.regularizer not in VALID_REGULARIZERS:
            raise ValueError(f"Choose regularizer from {VALID_REGULARIZERS}.")
        if self.initializer not in VALID_INITIALIZERS:
            raise ValueError(f"Choose initializer from {VALID_INITIALIZERS}.")
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(f"Choose activation from {VALID_ACTIVATIONS}.")

        if self.layer_widths is None:
            self.layer_widths = DEFAULT_LAYER_WIDTHS
        if self.learning_rates is None:
            self.learning_rates = DEFAULT_LEARNING_RATES
        if self.regularization_factors is None:
            self.regularizer_factors = DEFAULT_REGULARIZATION_FACTORS

        self.layer_widths = np.array(self.layer_widths)
        assert np.issubdtype(self.layer_widths.dtype, np.integer), \
            "`layer_widths` must be integers."
        assert np.all(self.layer_widths > 0), "`layer_widths` must be > 0."

        self.learning_rates = np.array(self.learning_rates)
        assert np.issubdtype(self.learning_rates.dtype, np.floating), \
            "`learning_rates` must be floats."
        assert np.all(self.learning_rates > 0), "`learning_rates` must be > 0."

    def _make_model(self, width=4096, learning_rate=0.01, reg_factor=0.001):
        """Create a SLFN model."""
        if self.regularizer is None:
            kernel_regularizer = None
        elif self.regularizer == "L1":
            kernel_regularizer = L1(l1=reg_factor)
        elif self.regularizer == "L2":
            kernel_regularizer = L2(l2=reg_factor)
        elif self.regularizer == "L1L2":
            kernel_regularizer = L1L2(l1=reg_factor, l2=reg_factor)
        else:
            raise ValueError(f"Unknown regularizer `{self.regularizer}`")

        def denselayer(width, activation):
            return tf.keras.layers.Dense(
                width, activation=activation,
                kernel_initializer=self.initializer,
                kernel_regularizer=kernel_regularizer)

        input_layer = tf.keras.Input(shape=(len(self.features),))
        output_layer = denselayer(len(self.targets), activation="linear")
        hidden_layer = denselayer(width=width, activation=self.activation)
        model = tf.keras.Sequential([input_layer, hidden_layer, output_layer])
        model.learning_rate = learning_rate
        return model

    def _model_builder(self, hyperparameters):
        """Builder of models for the hyperparameter tuning."""
        n_neurons = hyperparameters.Choice(
            'neurons', values=self.layer_widths)
        learning_rate = hyperparameters.Choice(
            'learning_rate', values=self.learning_rates)
        regularization_factor = hyperparameters.Choice(
            'regularization_factor', values=self.regularization_factors)

        model = self._make_model(width=n_neurons, learning_rate=learning_rate,
                                reg_factor=regularization_factor)
        # compile the model
        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_squared_error", "mean_absolute_error"])
        return model

    def tune(self, folder, project_name, objective="val_mean_absolute_error"):
        """Tune the model by performing the hyperparameter search."""
        self.tuner = keras_tuner.GridSearch(
            self._model_builder, objective=objective,
            directory=folder, project_name=project_name)
