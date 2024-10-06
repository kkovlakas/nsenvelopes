"""Tuning and training of single hidden layer feedforward neural networks."""


# Disable warning regarding resolving `keras` - this is a known pylint issue
# pylint: disable=E1101,E0401,E0611


import os
import json

import pandas as pd
import tensorflow as tf
from tensorflow.keras.regularizers import L1, L2, L1L2
from keras_tuner import GridSearch

from nsenvelopes.io import clear_folder


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
    def __init__(self, subsets, features, targets,
                 separate_holdout=True, layer_widths=None, learning_rates=None,
                 regularization_factors=None, regularizer="L1L2",
                 initializer="he_normal", activation="sigmoid", batch_size=64,
                 verbose=True):
        """Initialize the ModelArchitect.

        Parameters
        ----------
        subsets : dict
            A dictionary with keys `X_train`, `y_train`, `X_valid`, `y_valid`,
            corresponding to the features (`X_*`) and targets (`y_*`) in the
            training (`_train`) and validation datasets (`_valid`).
        features : list of str
            A list of column names of the tables from the subsets, indicating
            the features.
        targets : list of str
            A list of column names of the tables from the subsets, indicating
            the targets. If only one target, use a single-element list.
        separate_holdout : True
            If a `_holdout` subset exists in addition to `_test`. Default True.
        layer_widths : list of int
            The different choices for the number of units in the hidden layer.
        learning_rates : list of float
            The different choices for the learning rate.
        regularization_factors : list of float
            The different choices for the regularization factor.
        regularizer : str or None
            The type of kernel regularizer to be used.
            Choices are: L1, L2, L1L12 (default), and None (no regularization).
        initializer : str
            The name of the weight initializer. Default: "he_normal".
        activation : str
            The hidden layer's activation function. Default: "sigmoid".
        verbose : bool
            If True, report of various things.
        """
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
        self.batch_size = batch_size
        self.verbose = verbose
        self.tuner = None
        self._validity_check()

    def _say(self, *args, **kwargs):
        """Print the message if verbose is True."""
        if self.verbose:
            print(*args, **kwargs)

    def _validity_check(self):
        """Check the validity of property values."""
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
            self.regularization_factors = DEFAULT_REGULARIZATION_FACTORS

        if not isinstance(self.batch_size, int):
            raise ValueError("Batch size must be an integer.")
        if self.batch_size <= 1:
            raise ValueError("Batch size must be greater than 1.")

        check_positive_list(self.layer_widths, int, "layer_widths")
        check_positive_list(self.learning_rates, float, "learning_rates")
        check_positive_list(self.regularization_factors, float,
                            "regularization_factors")

    def _make_model(self, width=4096, learning_rate=0.01, reg_factor=0.001):
        """Make a model given the # of nuerons, learning rate and reg. factor.

        Parameters
        ----------
        width : int
            The number of neurons in the hidden layer, by default 4096.
        learning_rate : float
            The learning rate for the ADAM optimizer, by default 0.01.
        reg_factor : float
            The regularization factor, by default 0.001.

        Returns
        -------
        model
            The sequential TensorFlow model for a single-hidden layer network.

        """
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
            """Create a dense layer of a given `width` and `activation`."""
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

    def _hyper_model_builder(self, hyperparameters):
        """Builder of models for the hyperparameter tuning."""
        n_neurons = hyperparameters.Choice(
            'neurons', values=list(self.layer_widths))
        learning_rate = hyperparameters.Choice(
            'learning_rate', values=self.learning_rates)
        regularization_factor = hyperparameters.Choice(
            'regularization_factor', values=self.regularization_factors)

        return self.build_model(width=n_neurons, learning_rate=learning_rate,
                                reg_factor=regularization_factor)

    def build_model(self, width=4096, learning_rate=0.01, reg_factor=0.001):
        """Create and build a single hidden layer TensorFlow model."""
        model = self._make_model(
            width=width, learning_rate=learning_rate, reg_factor=reg_factor)
        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_squared_error", "mean_absolute_error"])
        return model

    def fit_model(self, model, epochs=100,
                  plot=True, export_path=None,
                  use_multiprocessing=True, n_cpus=11):
        """Fit a built model, and export it with its history."""
        self._say("Fitting the model...")
        model.fit(self.subsets["X_train"][self.features],
                  self.subsets["y_train"][self.targets],
                  epochs=epochs, batch_size=self.batch_size,
                  validation_data=[self.subsets["X_valid"][self.features],
                                   self.subsets["y_valid"][self.targets]],
                  verbose=int(self.verbose),
                  use_multiprocessing=use_multiprocessing,
                  workers=n_cpus)

        self._say("Clearing folder and saving model...")
        clear_folder(export_path)
        model.save(export_path)

        self._say("Exporting history...")
        history_path = os.path.join(export_path, "history.json")
        json.dump(model.history.history,
                  open(history_path, "w", encoding="utf-8"))

    def tune(self, folder, project_name, objective="val_mean_absolute_error",
             n_epochs=100, use_multiprocessing=True, n_cpus=11,
             early_stopping_monitor='val_loss', early_stopping_patience=10):
        """Tune the model and save the results in the given folder.

        Parameters
        ----------
        folder : str
            The folder holding hyperparameter tuning results.
        project_name : str
            Name of the subfolder for the current tuning project.
        objective : str,
            The hyperparam. score metric, by default "val_mean_absolute_error".
        n_epochs : int
            Maximum number of epochs during the search, by default 100.
        use_multiprocessing : bool
            Whether to use multiprocessing, by default True.
        n_cpus : int
            Number of CPUs if `use_multiprocessing` is True, by default 11.
        early_stopping_monitor : str
            The metric to use to decide on early stopping. Default 'val_loss'.
        early_stopping_patience : int
            Number of steps to wait without improvement, by default 10.
        """
        self.tuner = GridSearch(self._model_builder, objective=objective,
                                directory=folder, project_name=project_name)

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_monitor, patience=early_stopping_patience)

        self.tuner.search(
            self.subsets["X_train"][self.features],
            self.subsets["y_train"][self.targets],
            validation_data=[self.subsets["X_valid"][self.features],
                             self.subsets["y_valid"][self.targets]],
            epochs=n_epochs, batch_size=self.batch_size,
            use_multiprocessing=use_multiprocessing, workers=n_cpus,
            callbacks=[stop_early]
        )

    @staticmethod
    def get_tuner_results(directory, project_name):
        """Get the metrics from the hyperparameter tuning folder.

        Parameters
        ----------
        directory : _type_
            _description_
        project_name : _type_
            _description_

        Returns
        -------
        pandas DataFrame
            A dataframe containing the hyperparameter tuning results, namely,
            the IDs, hyperparameter choices, best step, score, as well as the
            training and validation metrics (loss, mean absolute error, mean
            squared error) of each trial.
        """
        model = ModelArchitect(subsets=None, features=None, targets=None)
        tuner = GridSearch(model._hyper_model_builder, objective=None,
                           directory=directory, project_name=project_name,
                           overwrite=False)
        trials = tuner.oracle.trials

        # use the first trial to get the column names for the results table
        a_trial = list(trials.values())[0]
        metrics = list(a_trial.metrics.metrics.keys())
        hypers = list(a_trial.hyperparameters.values.keys())
        columns = ["trial"] + hypers + ["best_step", "score"] + metrics
        results = {column: [] for column in columns}

        # iterate the trials and fill in the information
        for trial_id, trial in trials.items():
            results["trial"].append(trial_id)
            results["best_step"].append(trial.best_step)
            results["score"].append(trial.score)
            for hyper, hyper_value in trial.hyperparameters.values.items():
                results[hyper].append(hyper_value)
            for metric, metrics_data in trial.metrics.metrics.items():
                results[metric].append(metrics_data.get_best_value())

        return pd.DataFrame.from_dict(results)


def check_positive_list(lst, datatype, name):
    """Check if `lst` is a list of positive numbers of a given `datatype`."""
    if not isinstance(lst, list):
        raise ValueError(f"`{name}` must be a list.")
    for element in lst:
        if not isinstance(element, datatype) or element <= 0:
            raise ValueError(f"`{name}` must contain positive {datatype}s.")
