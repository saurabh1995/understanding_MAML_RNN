"""Spiking Legendre Memory Unit (SLMU) cells.

These were the 'most important block' of the notebook (cell 5) but are only referenced in
commented-out code there, so they are kept here for reference and are not part of the
model builder's cell registry. Usage::

    cell  = SLMUcell_LIF_thr(memory_d=10, order=10, theta=784,
                             hidden_cell=tf.keras.layers.SimpleRNNCell(units=208))
    layer = tf.keras.layers.RNN(cell, return_sequences=True)

The notebook had four ~250-line copies of the same class. They only differ in whether the
threshold is trainable and in what the cell emits, which is what the subclasses below set:

    class              threshold                       output
    SLMUcell           fixed (``thr`` argument)        spikes z
    SLMUcell_thr       trainable, per unit, U(.01,.09) spikes z
    SLMUcell_thr_V     trainable, per unit             tanh(v)
    SLMUcell_LIF_thr   trainable, per unit             z * tanh(v)

Changes w.r.t. the notebook: the unused ``DropoutRNNCellMixin`` (a Keras-internal import)
was dropped, ``hidden_cell`` is required (the ``hidden_cell=None`` branch referenced
attributes that never existed) and ``get_config`` no longer references the undefined
``recurrent_regularizer``.
"""

import numpy as np
import tensorflow as tf

from .surrogates import get_spike_function


class SLMUcell(tf.keras.layers.Layer):

    trainable_thr = False
    output_mode = "spikes"      # "spikes" | "tanh_v" | "spikes_tanh_v"

    def __init__(self, memory_d, order, theta, hidden_cell, thr=0.03, tau=20, dampening_factor=0.3,
                 kernel_initializer="glorot_uniform", kernel_regularizer=None, surrogate="fast_sigmoid",
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_cell is None:
            raise ValueError("SLMU cells require a hidden_cell")

        self.decay = tf.exp(-1 / tau)
        self.dampening_factor = dampening_factor
        self.thr = None if self.trainable_thr else thr
        self.surrogate = surrogate
        self.spike_function = get_spike_function(surrogate)

        self.memory_d = memory_d
        self.order = order
        self._init_theta = theta
        self.hidden_cell = hidden_cell
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel = None
        self.theta_inv = None
        self.A = None
        self.B = None

        if hasattr(self.hidden_cell, "state_size"):
            self.hidden_output_size = self.hidden_cell.output_size
            self.hidden_state_size = self.hidden_cell.state_size
        else:
            self.hidden_output_size = self.hidden_cell.units
            self.hidden_state_size = [self.hidden_cell.units]

        self.state_size = tf.nest.flatten(self.hidden_state_size) + [self.memory_d * self.order]
        self.output_size = self.hidden_output_size

    def theta(self):
        """Value of the ``theta`` parameter."""
        if self.built:
            return 1 / tf.keras.backend.get_value(self.theta_inv)
        return self._init_theta

    def _gen_AB(self):
        """Generates the (zero-order-hold discretised) LMU A and B matrices."""
        Q = np.arange(self.order, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R

        self._base_A = tf.constant(A.T, dtype=self.dtype)
        self._base_B = tf.constant(B.T, dtype=self.dtype)
        self.A, self.B = self._cont2discrete_zoh(self._base_A / self._init_theta,
                                                 self._base_B / self._init_theta)

    @staticmethod
    def _cont2discrete_zoh(A, B):
        # combine A/B and pad to make square matrix
        em_upper = tf.concat([A, B], axis=0)
        em = tf.pad(em_upper, [(0, 0), (0, B.shape[0])])
        ms = tf.linalg.expm(em)
        # slice A/B back out of combined matrix
        return ms[: A.shape[0], : A.shape[1]], ms[A.shape[0]:, : A.shape[1]]

    def build(self, input_shape):
        super().build(input_shape)
        enc_d = input_shape[-1]

        if self.kernel_initializer is not None:
            self.kernel = self.add_weight(name="kernel", shape=(enc_d, self.memory_d),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer)
        elif enc_d != self.memory_d:
            raise ValueError(f"For LMUCells with no input kernel, the input dimension ({enc_d})"
                             f" must equal `memory_d` ({self.memory_d}).")

        self.theta_inv = tf.constant(1 / self._init_theta, dtype=self.dtype)

        if not self.hidden_cell.built:
            with tf.name_scope(self.hidden_cell.name):
                self.hidden_cell.build((input_shape[0], self.memory_d * self.order))

        if self.trainable_thr:
            self.thr = tf.Variable(tf.random.uniform(shape=(self.hidden_output_size,), minval=0.01, maxval=0.09))

        self._gen_AB()

    def call(self, inputs, states, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        states = tf.nest.flatten(states)
        h = states[:-1]     # hidden cell state
        m = states[-1]      # LMU memory

        old_spikes = tf.cast(tf.greater((h[0] - self.thr) / self.thr, 0.), dtype=tf.float32)

        u = inputs
        if self.kernel is not None:
            u = tf.matmul(u, self.kernel, name="kernel_matmul")

        # memory update, separating memory/order dimensions
        m = tf.reshape(m, (-1, self.memory_d, self.order))
        u = tf.expand_dims(u, -1)
        m = tf.matmul(m, self.A) + tf.matmul(u, self.B)
        m = tf.reshape(m, (-1, self.memory_d * self.order))

        if hasattr(self.hidden_cell, "state_size"):
            o, h = self.hidden_cell(m, h, training=training)
        else:
            o = self.hidden_cell(m, training=training)

        o_new = o - self.thr * old_spikes
        new_z = self.spike_function((o_new - self.thr) / self.thr, self.dampening_factor)

        if self.output_mode == "spikes":
            out = new_z
        elif self.output_mode == "tanh_v":
            out = tf.math.tanh(o_new)
        else:
            out = new_z * tf.math.tanh(o_new)
        # o_new replaces the hidden cell's first state (it is read back as h[0] above);
        # any further hidden states (e.g. the LSTM carry) are passed through unchanged
        h = tf.nest.flatten(h)
        return out, [o_new] + h[1:] + [m]

    def get_config(self):
        config = super().get_config()
        config.update(memory_d=self.memory_d, order=self.order, theta=self._init_theta,
                      hidden_cell=tf.keras.layers.serialize(self.hidden_cell),
                      kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer,
                      surrogate=self.surrogate)
        return config

    @classmethod
    def from_config(cls, config):
        config["hidden_cell"] = tf.keras.layers.deserialize(config["hidden_cell"])
        return super().from_config(config)


class SLMUcell_thr(SLMUcell):
    trainable_thr = True


class SLMUcell_thr_V(SLMUcell):
    trainable_thr = True
    output_mode = "tanh_v"


class SLMUcell_LIF_thr(SLMUcell):
    trainable_thr = True
    output_mode = "spikes_tanh_v"
