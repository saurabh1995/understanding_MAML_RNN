"""Config-driven model builder.

The notebook had two definitions of the network: a subclassed ``TrainModel`` used for
MAML and a functional ``model_klassiche`` used for classical training, fine-tuning and
validation. Their weights were copied into each other with ``set_weights``, so they had
to stay in sync by hand. Here a single builder is used for every strategy.

Default architecture (= ``model_klassiche`` of the notebook)::

    Input(None, 5) -> 3 x RNN(HSNN(208, thr=1e-7)) -> tanh -> Dense(240) -> Dense(64) -> Dense(5)

Resulting ``model.layers`` order: Input, RNN..., [decoder RNN], Activation, Dense..., Dense(out).
"""

import copy

import tensorflow as tf

from .cells import CELLS


KERAS_CELLS = {
    "LSTM": tf.keras.layers.LSTMCell,
    "GRU": tf.keras.layers.GRUCell,
    "SimpleRNN": tf.keras.layers.SimpleRNNCell,
}

# arguments of the spiking cells that the Keras cells do not accept
SPIKING_ONLY_PARAMS = ("thr", "tau", "dampening_factor", "surrogate")


def make_cell(cell_type, units, params=None):
    params = dict(params or {})
    if cell_type in KERAS_CELLS:
        params = {k: v for k, v in params.items() if k not in SPIKING_ONLY_PARAMS}
        return KERAS_CELLS[cell_type](units, **params)
    try:
        return CELLS[cell_type](units, **params)
    except KeyError:
        raise ValueError(f"Unknown cell '{cell_type}'. Available: {sorted(CELLS) + sorted(KERAS_CELLS)}") from None


def build_model(model_cfg, seq_len=None):
    """Build an (uncompiled) functional Keras model from the ``model`` config section.

    ``seq_len`` is only needed when ``unroll: true`` (Keras can only unroll a known
    number of time steps).
    """
    cfg = copy.deepcopy(model_cfg)
    rnn = cfg["rnn"]
    unroll = cfg.get("unroll", False)
    if unroll and seq_len is None:
        raise ValueError("model.unroll=true requires a known sequence length")

    inputs = tf.keras.Input((seq_len if unroll else None, cfg["input_dim"]))
    x = inputs
    for _ in range(rnn["num_layers"]):
        cell = make_cell(rnn["cell"], rnn["units"], rnn.get("params"))
        x = tf.keras.layers.RNN(cell, return_sequences=True, unroll=unroll)(x)

    decoder = cfg.get("decoder")
    if decoder:
        cell = make_cell(decoder["cell"], decoder["units"], decoder.get("params"))
        x = tf.keras.layers.RNN(cell, return_sequences=True, unroll=unroll)(x)

    if cfg.get("post_rnn_activation"):
        x = tf.keras.layers.Activation(cfg["post_rnn_activation"])(x)

    for units in cfg.get("dense_units", []):
        x = tf.keras.layers.Dense(units, activation=cfg.get("dense_activation", "tanh"))(x)
    outputs = tf.keras.layers.Dense(cfg["output_dim"], activation=cfg.get("output_activation", "tanh"))(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def model_factory(model_cfg, seq_len=None):
    """Zero-argument builder, handy for strategies that need fresh copies of the model."""
    return lambda: build_model(model_cfg, seq_len)
