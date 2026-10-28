"""Hyperband search over the network size (notebook cells 11-14).

Searches the number/width of RNN layers (cell type from ``model.rnn.cell``) and of the
Dense layers. All other model settings come from the config.

    python tune.py --config configs/classical.yaml --set tuning.max_epochs=100
"""

import argparse

import tensorflow as tf

from maml_rnn.config import load_config
from maml_rnn.data import prepare_data
from maml_rnn.losses import my_loss
from maml_rnn.models import build_model
from maml_rnn.utils import set_seed, setup_gpu


def make_hypermodel(model_cfg, tuning, lr, seq_len=None):

    def model_builder(hp):
        cfg = dict(model_cfg)
        cfg["rnn"] = dict(model_cfg["rnn"])
        cfg["rnn"]["num_layers"] = hp.Int('SRNN_layers', *tuning["rnn_layers"], default=tuning["rnn_layers"][0])
        lo, hi, step = tuning["rnn_units"]
        cfg["rnn"]["units"] = hp.Int('SRNN_units', min_value=lo, max_value=hi, step=step)
        lo, hi, step = tuning["dense_units"]
        dense_units = hp.Int('units_dense', min_value=lo, max_value=hi, step=step)
        n_dense = hp.Int('Dense_layers', *tuning["dense_layers"], default=tuning["dense_layers"][0])
        cfg["dense_units"] = [dense_units] * n_dense

        model = build_model(cfg, seq_len)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=my_loss)
        return model

    return model_builder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    try:
        import keras_tuner as kt
    except ImportError:
        raise SystemExit("tune.py needs keras-tuner: pip install keras-tuner")

    cfg = load_config(args.config, args.overrides)
    tuning = cfg["tuning"]
    setup_gpu()
    set_seed(cfg.get("seed"))
    data = prepare_data(cfg["data"])

    tuner = kt.Hyperband(make_hypermodel(cfg["model"], tuning, cfg["strategies"]["classical"]["lr"], data.seq_len),
                         objective='loss', max_epochs=tuning["max_epochs"], factor=tuning["factor"],
                         directory=tuning["directory"], project_name=tuning["project_name"],
                         hyperband_iterations=tuning["hyperband_iterations"])
    tuner.search(data.train_x, data.train_y, batch_size=tuning["batch_size"], epochs=tuning["max_epochs"])

    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:", best_hp.values)


if __name__ == "__main__":
    main()
