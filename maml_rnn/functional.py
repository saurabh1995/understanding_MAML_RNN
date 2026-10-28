"""Evaluate a Keras model with arbitrary parameter *tensors* instead of its variables.

MAML needs the loss at the adapted parameters ``theta' = theta - lr * grad`` while keeping
``theta'`` a differentiable function of ``theta``. Assigning ``theta'`` to variables would
cut that dependency. The notebook solved this by building a fresh copy of the model for
every task (``copy_model``) and overwriting each layer's weight attribute
(``cell.input_weights``, ``dense.kernel``, ...) with a tensor, using hand-written
per-layer index bookkeeping (``k += 2`` / ``k += 3``).

``ShadowModel`` does the same generically and builds the copy only once:

* every trainable variable that a layer (or an RNN layer's cell) stores as an attribute is
  found automatically and mapped to its index in ``model.trainable_variables``;
* ``shadow(params, x)`` writes ``params[i]`` into those attributes and runs the model.

Note: once an attribute has been replaced by a tensor Keras stops tracking the variable,
so the shadow's own ``trainable_variables`` / ``set_weights`` must not be used afterwards.
The index mapping is therefore computed before the first substitution.
"""

import tensorflow as tf


def _weight_owners(model):
    """Yield ``(layer, owner, attr_name, variable)`` for every trainable weight attribute."""
    for layer in model.layers:
        for owner in (layer, getattr(layer, "cell", None)):
            if owner is None:
                continue
            for name, value in list(vars(owner).items()):
                if isinstance(value, tf.Variable) and value.trainable:
                    yield layer, owner, name, value


def param_layers(model):
    """Layer owning each entry of ``model.trainable_variables`` (same order)."""
    index = {v.ref(): i for i, v in enumerate(model.trainable_variables)}
    layers = [None] * len(index)
    for layer, _, _, var in _weight_owners(model):
        layers[index[var.ref()]] = layer
    missing = [model.trainable_variables[i].name for i, l in enumerate(layers) if l is None]
    if missing:
        raise RuntimeError(f"Could not locate the owning layer of: {missing}")
    return layers


class ShadowModel:

    def __init__(self, build_fn):
        self.model = build_fn()
        index = {v.ref(): i for i, v in enumerate(self.model.trainable_variables)}
        self.num_params = len(index)
        self._slots = [(owner, name, index[var.ref()]) for _, owner, name, var in _weight_owners(self.model)]
        if len({i for _, _, i in self._slots}) != self.num_params:
            raise RuntimeError("Not every trainable variable is stored as a layer attribute; "
                               "ShadowModel cannot substitute it.")

    def __call__(self, params, x, training=False):
        if len(params) != self.num_params:
            raise ValueError(f"Expected {self.num_params} parameter tensors, got {len(params)}")
        for owner, name, i in self._slots:
            setattr(owner, name, params[i])
        return self.model(x, training=training)
