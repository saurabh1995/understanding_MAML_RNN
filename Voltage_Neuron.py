import tensorflow as tf
import numpy as np

class VSN(tf.keras.layers.Layer):
    
    def __init__(self, num_neurons, thr=0.03, tau=20., dampening_factor=0.3):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.state_size = (num_neurons, num_neurons)
        self.decay = tf.exp(-1 / tau)

        self.dampening_factor = dampening_factor
        self.thr = thr

        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None


    def build(self, input_shape):
        n_in = input_shape[-1]
        n    = self.num_neurons
        
        rand_init = tf.keras.initializers.RandomNormal

        # define the input weight variable
        self.input_weights = self.add_weight(
            shape=(n_in,n),
            initializer=rand_init(stddev=1. / np.sqrt(n_in)),
            name='input_weights')
        
        # define the recurrent weight variable
        self.disconnect_mask = tf.cast(np.diag(np.ones(n, dtype=bool)), tf.bool)
        self.recurrent_weights = self.add_weight(
            shape=(n, n),
            initializer=rand_init(stddev=1. / np.sqrt(n)),
            name='recurrent_weights')
        
        self.thr = self.add_weight(
            shape=(n,),
            initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
            name='threshold_potential', trainable=True, constraint=tf.keras.constraints.NonNeg())

        self.decay = self.add_weight(
                                            shape=(n,), 
                                            initializer=tf.keras.initializers.Constant(0.9), 
                                            trainable=True,
                                            name='decay_factor', 
                                            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0)
                                        )
        

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
        d       = self.decay
        i_reset = - (self.thr) * old_spike
        new_v   = d * old_v + (1-d)*i_in + i_reset

        v_scaled   = (new_v - (self.thr)) / (self.thr)
        new_z      = SpikeFunction(v_scaled, self.dampening_factor)
        
        new_state = (new_v, new_z)
        
        return (new_v), new_state




@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):

    # This means: z = 1 if v > thr else 0
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        # This is where we overwrite the gradient
        # dy = dE/dz (total derivative) is the gradient back-propagated from the loss down to z(t)
        dE_dz = dy
        dz_dv_scaled = gaussian_surrogate(v_scaled)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad

def pseudo_derivative(v_scaled, dampening_factor):
    return tf.math.atan(tf.maximum(0.,1 - tf.abs(v_scaled))*dampening_factor)

def surrogate_gradient(v, gamma=1.0):
    return tf.maximum(0., 1 - tf.abs(v) / gamma)

def logistic_surrogate(v, beta=10.0):
    # Compute the sigmoid activation
    sig = tf.sigmoid(beta * v)
    # Its derivative: beta * sig * (1 - sig)
    return beta * sig * (1 - sig)

def gaussian_surrogate(v, gamma=1.0):
    # Returns e^(-0.5*(v/gamma)^2)
    return tf.exp(-0.5 * tf.square(v / gamma))


