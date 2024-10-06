"""Tuning and training of single hidden layer feedforward neural networks."""


# Disable warning regarding resolving `keras` - this is a known pylint issue
# pylint: disable=E1101,E0401,E0611


import pandas as pd
import tensorflow as tf
from tensorflow.keras.regularizers import L1, L2, L1L2
from keras_tuner import GridSearch


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
                 activation="sigmoid",
                 verbose=True):
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
        self.verbose = verbose

        self.tuner = None
        self.best_model = None
        self.selected_model = None

        self._validity_check()

    def _say(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

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
            self.regularization_factors = DEFAULT_REGULARIZATION_FACTORS

        if not is_positive_list(self.layer_widths, int):
            raise ValueError(
                "`layer_widths` must be a list of positive ints.")
        if not is_positive_list(self.learning_rates, float):
            raise ValueError(
                "`learning_rates` must be a list of positive floats.")
        if not is_positive_list(self.regularization_factors, float):
            raise ValueError(
                "`Regularization_factors` is not a list of positive floats.")

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
            'neurons', values=list(self.layer_widths))
        learning_rate = hyperparameters.Choice(
            'learning_rate', values=self.learning_rates)
        regularization_factor = hyperparameters.Choice(
            'regularization_factor', values=self.regularization_factors)

        model = self._make_model(width=n_neurons, learning_rate=learning_rate,
                                 reg_factor=regularization_factor)
        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_squared_error", "mean_absolute_error"])
        return model

    def tune(self, folder, project_name, objective="val_mean_absolute_error",
             n_epochs=100, batchsize=64, use_multiprocessing=True, n_cpus=11,
             early_stopping_monitor='val_loss', early_stopping_patience=10):
        """Tune the model by performing the hyperparameter search."""
        self.tuner = GridSearch(self._model_builder, objective=objective,
                                directory=folder, project_name=project_name)

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_monitor, patience=early_stopping_patience)

        x_train = self.subsets["X_train"][self.features]
        y_train = self.subsets["y_train"][self.targets]
        x_valid = self.subsets["X_valid"][self.features]
        y_valid = self.subsets["y_valid"][self.targets]

        self.tuner.search(
            x_train, y_train, validation_data=[x_valid, y_valid],
            epochs=n_epochs, batch_size=batchsize,
            use_multiprocessing=use_multiprocessing, workers=n_cpus,
            callbacks=[stop_early]
        )

    @staticmethod
    def get_tuner_results(directory, project_name):
        """Get the metrics from the hyperparameter tuning folder."""
        model = ModelArchitect(subsets=None, features=None, targets=None)
        tuner = GridSearch(model._model_builder, objective=None,
                           directory=directory, project_name=project_name,
                           overwrite=False)
        trials = tuner.oracle.trials

        a_trial = list(trials.values())[0]
        metrics = list(a_trial.metrics.metrics.keys())
        hypers = list(a_trial.hyperparameters.values.keys())
        columns = ["trial"] + hypers + ["best_step", "score"] + metrics

        results = {column: [] for column in columns}
        for trial_id, trial in trials.items():
            results["trial"].append(trial_id)
            results["best_step"].append(trial.best_step)
            results["score"].append(trial.score)
            for hyper, hyper_value in trial.hyperparameters.values.items():
                results[hyper].append(hyper_value)
            for metric, metrics_data in trial.metrics.metrics.items():
                results[metric].append(metrics_data.get_best_value())

        return pd.DataFrame.from_dict(results)


def is_positive_list(lst, datatype):
    """Return False if `lst` is not a Python list of a specific `datatype`."""
    if not isinstance(lst, list):
        return False
    for element in lst:
        if not isinstance(element, datatype) or element <= 0:
            return False
    return True
