"""Compare how fast differently pre-trained weights adapt to unseen sequences.

Notebook cells 37-59: for a few validation sequences, a fresh model is initialised with
each set of pre-trained weights (e.g. classical vs. meta-trained) and fine-tuned with
``model.fit`` on that single sequence. The loss curves are plotted against each other.

Example
-------
    python finetune_compare.py --config runs/classical_<ts>/config.yaml \
        --weights "Klassische Trainiertes Model=runs/classical_<ts>/weights.pkl" \
        --weights "Meta Trainiertes Model=runs/maml_v6_adapt_inner_outer_<ts>/weights.pkl" \
        --out plots/finetune --set finetune.sequence_indices=[0,2]

The model architecture comes from the config, so all weight files must belong to it.
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from maml_rnn.callbacks import EarlyStoppingByValue
from maml_rnn.config import load_config
from maml_rnn.data import prepare_data
from maml_rnn.losses import my_loss
from maml_rnn.models import model_factory
from maml_rnn.utils import load_weights, set_seed, setup_gpu


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML merged on top of configs/default.yaml (a run's config.yaml works)")
    parser.add_argument("--weights", action="append", required=True, metavar="LABEL=PATH",
                        help="pre-trained weights to compare (repeatable)")
    parser.add_argument("--out", default="plots/finetune", help="output directory for plots and curves")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def finetune(build_fn, weights, x, y, ft_cfg):
    model = build_fn()
    model.set_weights(weights)
    for idx in ft_cfg.get("freeze_layers") or []:
        model.layers[idx].trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(ft_cfg["lr"]), loss=my_loss)
    callbacks = [EarlyStoppingByValue("loss", ft_cfg["stop_loss"], verbose=1)] if ft_cfg.get("stop_loss") else []
    history = model.fit(x, y, epochs=ft_cfg["epochs"], batch_size=ft_cfg["batch_size"],
                        callbacks=callbacks, verbose=0)
    return history.history["loss"]


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    ft = cfg["finetune"]
    setup_gpu()
    set_seed(cfg.get("seed"))

    data = prepare_data(cfg["data"])
    build_fn = model_factory(cfg["model"], seq_len=data.seq_len)
    candidates = {}
    for item in args.weights:
        label, path = item.split("=", 1) if "=" in item else (os.path.basename(os.path.dirname(item)), item)
        candidates[label] = load_weights(path)

    n_val = len(data.val_x)
    indices = ft.get("sequence_indices") or [random.randint(0, n_val - 1) for _ in range(ft["num_sequences"])]
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({'font.size': 11})

    curves = {}
    for seq in indices:
        x, y = data.val_x[seq:seq + 1], data.val_y[seq:seq + 1]
        curves[seq] = {label: finetune(build_fn, w, x, y, ft) for label, w in candidates.items()}
        for label, losses in curves[seq].items():
            print(f"sequence {seq}: {label}: {losses[0]:.3e} -> {losses[-1]:.3e} ({len(losses)} epochs)")

        fig = plt.figure()
        for label, losses in curves[seq].items():
            plt.plot(losses, label=label)
        plt.xlabel('Self-learning steps')
        plt.ylabel('Validation loss')
        plt.legend()
        for fmt in ft["formats"]:
            fig.savefig(os.path.join(args.out, f"Meta_Klassich_seq_{seq}.{fmt}"), format=fmt)
        plt.close(fig)

    with open(os.path.join(args.out, "finetune_curves.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in curves.items()}, f, indent=2)
    print("Saved plots to", args.out)


if __name__ == "__main__":
    main()
