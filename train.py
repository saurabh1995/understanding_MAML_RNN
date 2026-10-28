"""Train the spiking RNN with any registered strategy.

Examples
--------
    python train.py --config configs/classical.yaml
    python train.py --config configs/maml.yaml
    python train.py --strategy maml_v3_meta_batch --set strategies.maml_v3_meta_batch.epochs=50
    python train.py --strategy classical --set model.rnn.cell=LSTM --set model.rnn.units=128

Each run writes to ``<output_dir>/<strategy>_<timestamp>/``:
    config.yaml   fully resolved configuration
    weights.pkl   pickled ``model.get_weights()``
    history.json  per-epoch curves, loss.png  their plot
"""

import argparse
import os

from maml_rnn.config import deep_update, load_config, save_config, strategy_config
from maml_rnn.data import prepare_data
from maml_rnn.losses import my_loss
from maml_rnn.models import model_factory
from maml_rnn.strategies import STRATEGIES, ALIASES, get_strategy, resolve_name
from maml_rnn.utils import (load_weights, make_run_dir, plot_history, save_history, save_weights,
                            set_seed, setup_gpu)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="experiment YAML merged on top of configs/default.yaml")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES) + sorted(ALIASES), help="overrides `strategy`")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set model.rnn.units=16 (repeatable)")
    parser.add_argument("--run-name", help="run directory prefix (default: strategy name)")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def evaluate(model, x, y):
    import tensorflow as tf
    return float(tf.reduce_mean(my_loss(y, model(x))))


def main():
    args = parse_args()
    overrides = list(args.overrides)
    if args.strategy:
        overrides.insert(0, f"strategy={args.strategy}")
    cfg = load_config(args.config, overrides)
    requested = cfg["strategy"]
    cfg["strategy"] = name = resolve_name(requested)
    if requested != name and requested in cfg.get("strategies", {}):
        # settings given under the alias (e.g. strategies.maml.epochs) apply to the resolved strategy
        alias_cfg = cfg["strategies"].pop(requested) or {}
        cfg["strategies"][name] = deep_update(cfg["strategies"].get(name, {}), alias_cfg)

    setup_gpu()
    set_seed(cfg.get("seed"))

    data = prepare_data(cfg["data"])
    print("Data:", data.summary())

    build_fn = model_factory(cfg["model"], seq_len=data.seq_len)
    model = build_fn()
    if cfg["training"].get("init_weights"):
        model.set_weights(load_weights(cfg["training"]["init_weights"]))
        print("Initialised from", cfg["training"]["init_weights"])
    if args.verbose:
        model.summary()

    strategy = get_strategy(name, strategy_config(cfg, name), build_fn, verbose=args.verbose)
    history = strategy.fit(model, data)

    results = {split: evaluate(model, x, y) for split, (x, y) in {
        "train": (data.train_x, data.train_y),
        "test": (data.test_x, data.test_y),
        "validation": (data.val_x, data.val_y),
    }.items()}
    print("Final MSE:", ", ".join(f"{k}={v:.3e}" for k, v in results.items()))

    run_dir = make_run_dir(cfg["output_dir"], args.run_name or name)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    save_weights(model, os.path.join(run_dir, "weights.pkl"))
    save_history({**history, **{f"final_{k}": [v] for k, v in results.items()}},
                 os.path.join(run_dir, "history.json"))
    plot_history(history, os.path.join(run_dir, "loss.png"), title=name)
    print("Saved run to", run_dir)


if __name__ == "__main__":
    main()
