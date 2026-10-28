"""Loading, scaling and splitting of the BVP stress-strain sequences.

Each sequence is one CSV pair ``strain_<id>.csv`` / ``stress_<id>.csv`` where ``<id>`` is
the concatenation of one value from each list in ``data.index_grid`` (the notebook's four
nested loops ``i in 2..3, j in 1..3, k in 1..3, l in {1, 5}`` -> 36 sequences).

Splits (as in the notebook):
    all  --(validation_size, validation_seed)-->  rest / validation
    rest --(test_size, test_seed)------------->  train / test

In MAML the training sequences are the tasks (support sets), the test split is the query
set and the validation split is used for monitoring and fine-tuning experiments.

Note: like the notebook, the min-max scalers are fitted on *all* sequences before
splitting.
"""

import itertools
import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SequenceData:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler

    @property
    def seq_len(self):
        return self.train_x.shape[1]

    def summary(self):
        return (f"train {self.train_x.shape} -> {self.train_y.shape}, "
                f"test {self.test_x.shape}, validation {self.val_x.shape}")


def load_sequences(data_cfg):
    """Return raw ``(inputs, outputs)`` arrays of shape ``(n_seq, n_steps, n_features)``."""
    root = os.path.expanduser(data_cfg["root"])
    columns = data_cfg["columns"]
    inputs, outputs = [], []
    for ids in itertools.product(*data_cfg["index_grid"]):
        seq_id = "".join(str(i) for i in ids)
        a = np.genfromtxt(os.path.join(root, f"{data_cfg['input_prefix']}_{seq_id}.csv"))
        b = np.genfromtxt(os.path.join(root, f"{data_cfg['output_prefix']}_{seq_id}.csv"))
        inputs.append(a[:, columns])
        outputs.append(b[:, columns])
    return np.array(inputs), np.array(outputs)


def _fit_scale(arr, feature_range):
    scaler = MinMaxScaler(feature_range=tuple(feature_range))
    flat = arr.reshape(-1, arr.shape[-1])
    return scaler.fit_transform(flat).reshape(arr.shape), scaler


def prepare_data(data_cfg):
    inputs, outputs = load_sequences(data_cfg)
    x, x_scaler = _fit_scale(inputs, data_cfg["feature_range"])
    y, y_scaler = _fit_scale(outputs, data_cfg["feature_range"])

    rest_x, val_x, rest_y, val_y = train_test_split(
        x, y, test_size=data_cfg["validation_size"], random_state=data_cfg["validation_seed"])
    train_x, test_x, train_y, test_y = train_test_split(
        rest_x, rest_y, test_size=data_cfg["test_size"], random_state=data_cfg["test_seed"])

    f32 = lambda a: a.astype(np.float32)
    return SequenceData(f32(train_x), f32(train_y), f32(test_x), f32(test_y), f32(val_x), f32(val_y),
                        x_scaler, y_scaler)


def make_dataset(x, y, batch_size, shuffle_buffer=1000, seed=None):
    ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x), tf.convert_to_tensor(y)))
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    return ds.batch(batch_size)
