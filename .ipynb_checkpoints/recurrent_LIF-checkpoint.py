import tensorflow as tf
import numpy as np


class RSNN(tf.keras.layers.Layer):
    
    def __init__(self, num_neurons, thr=0.03, tau=20., dampening_factor=0.3):
        super(RSNN, self).__init__()
        
        self.num_neurons       = num_neurons
        self.state_size        = (num_neurons, num_neurons)
        self.decay             = tf.exp(-1 / tau)

        self.dampening_factor  = dampening_factor
        self.thr               = thr

        self.input_weights     = None
        self.recurrent_weights = None
        self.disconnect_mask   = None


    def build(self, input_shape):
        
        n_in = input_shape[-1]
        n    = self.num_neurons
        
        rand_init = tf.keras.initializers.RandomNormal

        # define the input weight variable
        self.input_weights = self.add_weight(
            shape=(n_in,n),
            initializer=rand_init(stddev=1. / np.sqrt(n_in)),
            name='input_weights', trainable=True)
        
        # define the recurrent weight variable
        self.disconnect_mask = tf.cast(np.diag(np.ones(n, dtype=bool)), tf.bool)
        self.recurrent_weights = self.add_weight(shape=(n, n),
            initializer=rand_init(stddev=1. / np.sqrt(n)),
            name='recurrent_weights', trainable=True)
        
        super().build(input_shape)

    def get_recurrent_weights(self):
        
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)
        
        return w_rec

    def call(self, inputs, states, constants=None):
        
        old_v     = states[0]
        old_spike = states[1]

        # compute the input currents
        w_rec = self.get_recurrent_weights()
        i_in = inputs @ self.input_weights + old_spike @ w_rec
      
        # update the voltage
        d = self.decay
        i_reset = - self.thr * old_spike
        new_v = d * old_v + (1-d)*i_in + i_reset

        v_scaled   = (new_v - self.thr) / self.thr
        new_z      = SpikeFunction(v_scaled, self.dampening_factor)
        
        
        new_state = (new_v, new_z)
        return new_z,new_state



def pseudo_derivative(v_scaled, dampening_factor):
    
    return tf.math.atan(v_scaled * dampening_factor)


# @tf.custom_gradient
# def SpikeFunction(v_scaled, dampening_factor):

#     # This means: z = 1 if v > thr else 0
#     z_ = tf.greater(v_scaled, 0.)
#     z_ = tf.cast(z_, dtype=tf.float32)

#     def grad(dy):
#         # This is where we overwrite the gradient
#         # dy = dE/dz (total derivative) is the gradient back-propagated from the loss down to z(t)
#         dE_dz = dy
#         dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
#         dE_dv_scaled = dE_dz * dz_dv_scaled
#         #print(dz_dv_scaled)
#         return [dE_dv_scaled, tf.zeros_like(dampening_factor)]

#     return tf.identity(z_, name="SpikeFunction"), grad

# @tf.custom_gradient
# def SpikeFunction(v_scaled, dampening_factor, clip_norm=1.0):
#     # This means: z = 1 if v > 0 else 0
#     z_ = tf.greater(v_scaled, 0.)
#     z_ = tf.cast(z_, dtype=tf.float32)

#     def grad(dy):
#         # Backpropagate the gradient
#         dE_dz = dy
#         dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
#         dE_dv_scaled = dE_dz * dz_dv_scaled

#         # Clip gradients by norm (entire gradient vector)
#         dE_dv_scaled = tf.clip_by_norm(dE_dv_scaled, clip_norm)

#         # Return the clipped gradients
#         return [dE_dv_scaled, tf.zeros_like(dampening_factor)]

#     return tf.identity(z_, name="SpikeFunction"), grad

@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor, clip_norm_threshold=1.0):
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
        
        return [dE_dv_scaled_new, tf.zeros_like(dampening_factor)]

        

    return tf.identity(z_, name="SpikeFunction"), grad
