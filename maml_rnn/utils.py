import json
import os
import pickle
import random
from datetime import datetime

import numpy as np
import tensorflow as tf


def setup_gpu():
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_weights(model, path):
    """Pickle ``model.get_weights()`` (the format used in the notebook)."""
    with open(path, "wb") as f:
        pickle.dump(model.get_weights(), f)


def load_weights(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_run_dir(output_dir, name):
    path = os.path.join(output_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(path, exist_ok=True)
    return path


def save_history(history, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)


def plot_history(history, path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for key, values in history.items():
        if values:
            plt.semilogy(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if title:
        plt.title(title)
    plt.legend()
    fig.savefig(path)
    plt.close(fig)
