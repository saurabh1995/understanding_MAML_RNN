import tensorflow as tf
import numpy as np

class GSN(tf.keras.layers.Layer):
    def __init__(self, num_neurons, thr=0.03, tau=20., dampening_factor=0.3, num_bits=8):
        super().__init__()
        self.num_neurons = num_neurons
        self.state_size = (num_neurons, num_neurons)
        self.decay = tf.exp(-1 / tau)
        self.dampening_factor = dampening_factor
        self.thr = thr
        self.num_bits = num_bits
        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None
        self.v_scale = None  # Scaling factor for membrane potential

    def build(self, input_shape):
        n_in = input_shape[-1]
        n = self.num_neurons
        rand_init = tf.keras.initializers.RandomNormal

        # Initialize input weights
        self.input_weights = self.add_weight(
            shape=(n_in, n),
            initializer=rand_init(stddev=1. / np.sqrt(n_in)),
            name='input_weights'
        )

        # Initialize recurrent weights
        self.disconnect_mask = tf.cast(np.diag(np.ones(n, dtype=bool)), tf.bool)
        self.recurrent_weights = self.add_weight(
            shape=(n, n),
            initializer=rand_init(stddev=1. / np.sqrt(n)),
            name='recurrent_weights'
        )

        # Compute scaling factors
        self.compute_scaling_factors()

        # Quantize weights
        self.quantize_weights()

        super().build(input_shape)

    def compute_scaling_factors(self):
        # Determine the maximum representable integer value based on num_bits
        max_int_value = 2 ** (self.num_bits - 1) - 1

        # Compute scaling factor for input weights
        max_weight = tf.reduce_max(tf.abs(self.input_weights))
        self.input_scale = max_int_value / max_weight

        # Compute scaling factor for recurrent weights
        max_weight_rec = tf.reduce_max(tf.abs(self.recurrent_weights))
        self.recurrent_scale = max_int_value / max_weight_rec

        # Set a predefined scaling factor for membrane potential
        # This should be chosen based on the expected range of membrane potentials
        self.v_scale = 100  # Example value; adjust based on your network dynamics

    def quantize_weights(self):
        # Apply quantization to input weights
        self.input_weights.assign(tf.round(self.input_weights * self.input_scale))

        # Apply quantization to recurrent weights
        self.recurrent_weights.assign(tf.round(self.recurrent_weights * self.recurrent_scale))

    def get_recurrent_weights(self):
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)
        return w_rec

    def call(self, inputs, states):

        # Compute scaling factors
        self.compute_scaling_factors()

        # Quantize weights
        self.quantize_weights()

        
        old_v = states[0]
        old_spike = states[1]

        # Compute input currents
        w_rec = self.get_recurrent_weights()
        i_in = inputs @ self.input_weights + old_spike @ w_rec

        # Update the voltage
        d = self.decay
        i_reset = - self.thr * old_spike
        new_v = d * old_v + (1 - d) * i_in + i_reset

        # Scale and quantize the membrane potential
        new_v_scaled = tf.round(new_v * self.v_scale)

        # Determine spiking output
        int_thr = int(self.thr * self.v_scale)
        v_scaled = new_v_scaled - int_thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)

        new_state = (new_v_scaled, new_z*new_v_scaled)
        return new_z*new_v_scaled, new_state

@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):

    # This means: z = 1 if v > thr else 0
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.int32)

    def grad(dy):
        # This is where we overwrite the gradient
        # dy = dE/dz (total derivative) is the gradient back-propagated from the loss down to z(t)
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad


def pseudo_derivative(v_scaled, dampening_factor):
    return tf.math.atan(tf.maximum(0.,1 - tf.abs(v_scaled))*dampening_factor)

