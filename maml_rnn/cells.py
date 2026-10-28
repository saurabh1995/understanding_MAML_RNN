"""Recurrent spiking cells, to be wrapped in ``tf.keras.layers.RNN``.

All cells share the same leaky-integrate dynamics

    i_in  = x_t @ W_in + r_{t-1} @ W_rec        (W_rec without self-connections)
    v_t   = d * v_{t-1} + (1 - d) * i_in + i_reset,   d = exp(-1 / tau)
    z_t   = H((v_t - thr) / thr)                  (spike, surrogate gradient)

and differ in the recurrent signal r, the reset and what they emit:

    cell       recurrent r   reset          output              origin
    HSNN       spikes z      -thr * z       z * v  (hybrid)     Hybrid_Spiking_Neuron.py
    RSNN       spikes z      -thr * z       z                   recurrent_LIF.py
    RSNN_vthr  spikes z      -thr * z       tanh(z * v)         notebook, trainable threshold
    DSNN       voltage v     -thr (const.)  v  (decoder)        decoding.py

The MAML strategies replace ``input_weights`` / ``recurrent_weights`` (and ``thr`` for
RSNN_vthr) by plain tensors on a shadow model, so ``call`` must always read the weights
through these attributes.
"""

import numpy as np
import tensorflow as tf

from .surrogates import get_spike_function


class SpikingCell(tf.keras.layers.Layer):
    """Base class: weight creation and the shared leaky integration."""

    default_surrogate = "fast_sigmoid"

    def __init__(self, num_neurons, thr=0.03, tau=20., dampening_factor=0.3, surrogate=None, **kwargs):
        super().__init__(**kwargs)
        self.num_neurons = num_neurons
        self.state_size = (num_neurons, num_neurons)    # (voltage, spikes)
        self.tau = tau
        self.decay = tf.exp(-1 / tau)
        self.dampening_factor = dampening_factor
        self.thr = thr
        self.surrogate = surrogate or self.default_surrogate
        self.spike_function = get_spike_function(self.surrogate)

        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None

    def build(self, input_shape):
        n_in = input_shape[-1]
        n = self.num_neurons
        rand_init = tf.keras.initializers.RandomNormal

        self.input_weights = self.add_weight(
            shape=(n_in, n), initializer=rand_init(stddev=1. / np.sqrt(n_in)), name='input_weights')

        # no self-connections in the recurrent weights
        self.disconnect_mask = tf.cast(np.diag(np.ones(n, dtype=bool)), tf.bool)
        self.recurrent_weights = self.add_weight(
            shape=(n, n), initializer=rand_init(stddev=1. / np.sqrt(n)), name='recurrent_weights')

        super().build(input_shape)

    def get_recurrent_weights(self):
        return tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)

    def spike(self, v):
        v_scaled = (v - self.thr) / self.thr
        return self.spike_function(v_scaled, self.dampening_factor)

    def get_config(self):
        config = super().get_config()
        config.update(num_neurons=self.num_neurons, thr=float(self.thr) if np.isscalar(self.thr) else None,
                      tau=self.tau, dampening_factor=self.dampening_factor, surrogate=self.surrogate)
        return config


class HSNN(SpikingCell):
    """Hybrid spiking neuron: emits the spike-gated membrane voltage ``z * v``."""

    default_surrogate = "sigmoid"

    def call(self, inputs, states):
        old_v, old_spike = states[0], states[1]
        d = self.decay

        i_in = inputs @ self.input_weights + old_spike @ self.get_recurrent_weights()
        new_v = d * old_v + (1 - d) * i_in - self.thr * old_spike
        new_z = self.spike(new_v)

        return new_z * new_v, (new_v, new_z)


class RSNN(SpikingCell):
    """Recurrent LIF neuron: emits binary spikes."""

    default_surrogate = "atan_normalized"

    def call(self, inputs, states):
        old_v, old_spike = states[0], states[1]
        d = self.decay

        i_in = inputs @ self.input_weights + old_spike @ self.get_recurrent_weights()
        new_v = d * old_v + (1 - d) * i_in - self.thr * old_spike
        new_z = self.spike(new_v)

        return new_z, (new_v, new_z)


class RSNN_vthr(SpikingCell):
    """Recurrent LIF neuron with a trainable (scalar) threshold; emits ``tanh(z * v)``.

    As in the notebook, the threshold is initialised uniformly in [-10, 10]; the
    ``thr`` constructor argument is ignored.
    """

    def build(self, input_shape):
        super().build(input_shape)
        self.thr = self.add_weight(
            shape=(), initializer=tf.keras.initializers.RandomUniform(-10., 10.), name='thr')

    def call(self, inputs, states):
        old_v, old_spike = states[0], states[1]
        d = self.decay

        i_in = inputs @ self.input_weights + old_spike @ self.get_recurrent_weights()
        new_v = d * old_v + (1 - d) * i_in - self.thr * old_spike
        new_z = self.spike(new_v)

        return tf.math.tanh(new_z * new_v), (new_v, new_z)


class DSNN(SpikingCell):
    """Decoding neuron: voltage-recurrent, constant reset, emits the membrane voltage."""

    def call(self, inputs, states):
        old_v = states[0]
        d = self.decay

        i_in = inputs @ self.input_weights + old_v @ self.get_recurrent_weights()
        new_v = d * old_v + (1 - d) * i_in - self.thr
        new_z = self.spike(new_v)

        return new_v, (new_v, new_z)


CELLS = {
    "HSNN": HSNN,
    "RSNN": RSNN,
    "RSNN_vthr": RSNN_vthr,
    "DSNN": DSNN,
}
