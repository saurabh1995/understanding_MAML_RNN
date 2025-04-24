import tensorflow as tf
import numpy as np

@tf.custom_gradient
def quantize_with_ste(x, scale):
    # Forward pass: quantize the input
    y = tf.clip_by_value(tf.round(x * scale), -128., 127.)
    
    # Backward pass: straight-through estimator
    def grad(dy):
        return dy, tf.zeros_like(scale)
    
    return y, grad

@tf.custom_gradient
def quantize_with_V(x, scale):
    # Forward pass: quantize the input

    max_int = 2**31 - 1
    
    y = tf.clip_by_value(tf.round(x * scale), -float(128), float(127))

    
    # Backward pass: straight-through estimator
    def grad(dy):
        return dy, tf.zeros_like(scale)
    
    return y, grad

@tf.custom_gradient
def SpikeFunction_gradnorm(v_scaled, dampening_factor, clip_norm_threshold=1.0):
    # This means: z = 1 if v > 0 else 0
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        # Backpropagate the gradient
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled
        #print(dE_dv_scaled)
        # Calculate the norm of the gradient
        grad_norm = tf.linalg.norm(dE_dv_scaled)

        # Dynamically clip the gradient based on its norm
        #if grad_norm > clip_norm_threshold:
            # Scale the gradient to the threshold if the norm is larger than the threshold
        
        dE_dv_scaled_new = dE_dv_scaled*(clip_norm_threshold / grad_norm)
        print(grad_norm)
        
        return [dE_dv_scaled_new, tf.zeros_like(dampening_factor)]

        

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def SpikeFunction_clip(v_scaled, dampening_factor, clip_value=1.0):
    # Forward pass: spike if v_scaled > 0
    z_ = tf.cast(tf.greater(v_scaled, 0.0), dtype=tf.float32)

    def grad(dy):
        # Gradient from next layer
        dE_dz = dy

        # Surrogate gradient for spike function
        dz_dv_scaled = pseudo_derivative_(v_scaled, dampening_factor)

        # Chain rule
        dE_dv_scaled = dE_dz * dz_dv_scaled

        # Clip gradient only if its norm exceeds the threshold
        dE_dv_scaled_clipped = tf.clip_by_value(dE_dv_scaled, -clip_value, clip_value)
        print(dz_dv_scaled)
        # No gradient for dampening_factor
        return [dE_dv_scaled_clipped, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction_clip"), grad

@tf.custom_gradient
def SpikeFunction_clip(v_scaled, dampening_factor, clip_value=1.0):
    # Forward pass: spike if v_scaled > 0
    z_ = tf.cast(tf.greater(v_scaled, 0.0), dtype=tf.float32)

    def grad(dy):
        # Gradient from next layer
        dE_dz = tf.clip_by_value(dy, -clip_value, clip_value)

        # Surrogate gradient for spike function
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        # Chain rule
        dE_dv_scaled = dE_dz * dz_dv_scaled

        # Clip gradient only if its norm exceeds the threshold
        dE_dv_scaled_clipped = tf.clip_by_value(dE_dv_scaled, -clip_value, clip_value)
        #print(dE_dz)
        # No gradient for dampening_factor
        return [dE_dv_scaled_clipped, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction_clip"), grad

#............................................................................................................................................
def pseudo_derivative(v_scaled, dampening_factor):
    return tf.math.atan(tf.maximum(0.,150. - tf.abs(v_scaled))*dampening_factor)    
#............................................................................................................................................
def pseudo_derivative_fs(v_scaled, dampening_factor):
    x = v_scaled * dampening_factor
    return dampening_factor / (1.0 + tf.abs(x))**2
#............................................................................................................................................
def pseudo_derivative_(v_scaled, dampening_factor):
    return tf.math.atan((v_scaled)*dampening_factor)  



class GSN(tf.keras.layers.Layer):
    def __init__(self, num_neurons, thr=0.03, tau=20., dampening_factor=0.3, num_bits=8, v_scale=10):
        super(GSN, self).__init__()
        self.num_neurons = num_neurons
        self.state_size = (num_neurons, num_neurons)
        self.continuous_decay = tf.exp(-1 / tau)  # Continuous decay value
        self.dampening_factor = dampening_factor
        self.thr = thr
        self.num_bits = num_bits
        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None
        self.decay = None
        self.quantized_decay = None
        self.v_scale = v_scale  # Scaling factor for membrane potential

    def build(self, input_shape):
        n_in = input_shape[-1]
        n = self.num_neurons
        rand_init = tf.keras.initializers.RandomNormal

        # Initialize input weights
        self.input_weights = self.add_weight(
            shape=(n_in, n),
            initializer=rand_init(stddev=1. / np.sqrt(n_in)),
            name='input_weights', trainable=True
        )

        # Initialize recurrent weights
        self.disconnect_mask = tf.cast(np.diag(np.ones(n, dtype=bool)), tf.bool)
        self.recurrent_weights = self.add_weight(
            shape=(n, n),
            initializer=rand_init(stddev=1. / np.sqrt(n)),
            name='recurrent_weights', trainable=True
        )

        # Compute scaling factors
        self.compute_scaling_factors()

        # Quantize decay
        self.quantize_decay()

        super().build(input_shape)

    def compute_scaling_factors(self):
        # Determine the maximum representable integer value based on num_bits
        max_int_value = 2 ** (self.num_bits - 1) - 1

        # Compute scaling factor for input weights
        max_weight = tf.reduce_max(tf.abs(self.input_weights))
        self.input_scale = max_int_value / tf.maximum(max_weight, 1e-6)  # Avoid division by zero

        # Compute scaling factor for recurrent weights
        max_weight_rec = tf.reduce_max(tf.abs(self.recurrent_weights))
        self.recurrent_scale = max_int_value / tf.maximum(max_weight_rec, 1e-6)  # Avoid division by zero

        # Set a predefined scaling factor for membrane potential
        self.v_scale = 10.  # Example value; adjust based on your network dynamics

    def quantize_decay(self):
        # Loihi 2 uses 12 bits for decay parameters
        max_int_value = 2 ** 12 - 1  # 4095

        # Scale and quantize the continuous decay value
        scaled_decay = self.continuous_decay * max_int_value
        self.quantized_decay = tf.round(scaled_decay)

        # Convert back to continuous domain
        self.decay = self.quantized_decay / max_int_value

        

    def get_recurrent_weights(self):
        # Apply disconnect mask to recurrent weights
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)
        return w_rec

    def call(self, inputs, states, constants=None):
        old_v = states[0]
        old_spike = states[1]

        # @tf.custom_gradient
        # def quantize_with_ste(x, scale):
        #     # Forward pass: quantize the input
        #     y = tf.round(x * scale)
            
        #     # Backward pass: straight-through estimator
        #     def grad(dy):
        #         return dy, tf.zeros_like(scale)
            
        #     return y, grad
        
        # @tf.custom_gradient
        # def quantize_with_V(x, scale):
        #     # Forward pass: quantize the input
        #     y = tf.clip_by_value(tf.round(x * scale), -1000, 1000)
            
        #     # Backward pass: straight-through estimator
        #     def grad(dy):
        #         return dy, tf.zeros_like(scale)
            
        #     return y, grad
        
        # Dynamically compute quantized weights with STE
        quantized_input_weights     = quantize_with_ste(self.input_weights, self.input_scale)
        quantized_recurrent_weights = quantize_with_ste(self.recurrent_weights, self.recurrent_scale)
        
        # Apply the disconnect mask to the recurrent weights
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(quantized_recurrent_weights), quantized_recurrent_weights)

        # Compute input currents
        i_in = inputs @ quantized_input_weights + old_spike @ w_rec

        # Update the voltage
        d = self.quantized_decay
        
        int_thr = int(self.thr * self.v_scale)
        i_reset = - int_thr * old_spike
        new_v   = d * old_v + (1 - d) * i_in + i_reset

        # Scale and quantize the membrane potential
        new_v_scaled = quantize_with_V(new_v, self.v_scale)

        # Determine spiking output
        
        v_scaled = (new_v_scaled - int_thr)/int_thr
        new_z = SpikeFunction_clip(v_scaled, self.dampening_factor)

        new_state = (new_v_scaled, new_z*new_v_scaled)

        
        return new_z*new_v_scaled, new_state



