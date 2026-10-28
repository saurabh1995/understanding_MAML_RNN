"""Heaviside spike function with a selectable surrogate gradient.

The forward pass is always ``z = 1 if v_scaled > 0 else 0``. Only the backward pass
differs between the surrogates. The different notebooks/modules of this repo each used a
different one, so every cell keeps its original surrogate as its default:

    fast_sigmoid     d / (1 + |x|)^2                 (decoding.py -> DSNN, SLMU cells)
    sigmoid          d * s(x) * (1 - s(x))           (Hybrid_Spiking_Neuron.py -> HSNN)
    triangle_atan    atan(max(0, 1 - |x|) * d)       (`pseudo_derivative` in the old modules)
    atan_normalized  atan(x * d), rescaled to norm 1 (recurrent_LIF.py -> RSNN)
"""

import tensorflow as tf


def pseudo_derivative_fast_sigmoid(x, dampening_factor):
    return dampening_factor / tf.square(1.0 + tf.abs(x))


def pseudo_derivative_sigmoid(x, dampening_factor):
    s = tf.sigmoid(x)
    return dampening_factor * s * (1.0 - s)


def pseudo_derivative_triangle_atan(x, dampening_factor):
    return tf.math.atan(tf.maximum(0., 1 - tf.abs(x)) * dampening_factor)


def pseudo_derivative_atan(x, dampening_factor):
    return tf.math.atan(x * dampening_factor)


def _make_spike_function(pseudo_derivative, normalize=False, clip_norm=1.0):

    @tf.custom_gradient
    def spike_function(v_scaled, dampening_factor):
        z = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

        def grad(dy):
            # dy = dE/dz is the gradient back-propagated from the loss down to z(t);
            # the Heaviside derivative is replaced by the pseudo-derivative.
            dE_dv_scaled = dy * pseudo_derivative(v_scaled, dampening_factor)
            if normalize:
                # divide_no_nan: a zero gradient stays zero instead of becoming NaN
                dE_dv_scaled = dE_dv_scaled * tf.math.divide_no_nan(clip_norm, tf.linalg.norm(dE_dv_scaled))
            return [dE_dv_scaled, tf.zeros_like(dampening_factor)]

        return tf.identity(z, name="SpikeFunction"), grad

    return spike_function


SPIKE_FUNCTIONS = {
    "fast_sigmoid": _make_spike_function(pseudo_derivative_fast_sigmoid),
    "sigmoid": _make_spike_function(pseudo_derivative_sigmoid),
    "triangle_atan": _make_spike_function(pseudo_derivative_triangle_atan),
    "atan_normalized": _make_spike_function(pseudo_derivative_atan, normalize=True),
}


def get_spike_function(name):
    try:
        return SPIKE_FUNCTIONS[name]
    except KeyError:
        raise ValueError(f"Unknown surrogate '{name}'. Available: {sorted(SPIKE_FUNCTIONS)}") from None
