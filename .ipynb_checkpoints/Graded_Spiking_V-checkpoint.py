import tensorflow as tf
import numpy as np

@tf.custom_gradient
def quantize_with_ste(x):
 
    min_val  = tf.reduce_min(x)
    max_val  = tf.reduce_max(x)
    
    y = tf.quantization.fake_quant_with_min_max_vars(
        x, 
        min = min_val, 
        max = max_val, 
        num_bits=8,  narrow_range=True
    )

    
    
    def grad(dy):
        
        return dy
        
    return y, grad

@tf.custom_gradient
def quantize_with_V(x, min_val=-1.0, max_val=1.0):


    #print(scaled_x)
    # Step 2: Compute dynamic min and max values (based on signal range)
    val_min = tf.reduce_min(x)
    val_max = tf.reduce_max(x)
    
    y = tf.quantization.fake_quant_with_min_max_vars(x, min=val_min, max=val_max, num_bits=16)

    #print(y)
    
    
    
    def grad(dy):
        
        return dy
        
    return y, grad


@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):

    # This means: z = 1 if v > thr else 0
    z_ = tf.where(v_scaled > 0., tf.clip_by_value(tf.round(v_scaled), 0, 127), tf.zeros_like(v_scaled))
    #print(v_scaled)
    #z_ = tf.cast(z_, dtype=tf.float32)

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


#.............................main layer..............................................................................................................

class GSN(tf.keras.layers.Layer):
    
    def __init__(self, num_neurons, thr=0.01, tau=21., dampening_factor=0.30):
        super(GSN, self).__init__()
        self.num_neurons = num_neurons
        self.state_size = (num_neurons, num_neurons)
        self.decay = tf.exp(-1 / tau)  # Continuous decay value
        self.dampening_factor = dampening_factor
        self.thr = thr
        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None 

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


        # Quantize decay
        self.quantize_decay()

        super().build(input_shape)



    def quantize_decay(self):
        # Loihi 2 uses 12 bits for decay parameters
        max_value = 1
        min_value = 0


        # Convert back to continuous domain
        self.decay = tf.quantization.fake_quant_with_min_max_vars(
        inputs=self.decay,
        min=min_value,
        max=max_value,
        num_bits=12,
        narrow_range=True  # usually False for general use
    )

        

    def get_recurrent_weights(self):
        # Apply disconnect mask to recurrent weights
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)
        return w_rec

    def call(self, inputs, states, constants=None):
        
        old_v     = states[0]
        old_spike = states[1]
        #scale_thr = 1000.00
        #quantized_thr = tf.cast(tf.round(self.thr * scale_thr), tf.float32) / scale_thr
        
        
        # Dynamically compute quantized weights with STE
        quantized_input_weights     = quantize_with_ste(self.input_weights)
        quantized_recurrent_weights = quantize_with_ste(self.recurrent_weights)
        
        
        # Apply the disconnect mask to the recurrent weights
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(quantized_recurrent_weights), quantized_recurrent_weights)


        # Compute input currents
        i_in = inputs @ quantized_input_weights + old_spike @ w_rec

        # Update the voltage
        d   = self.decay
        
        
        i_reset = - self.thr * old_spike
        new_v   = d * old_v + (1 - d) * i_in + i_reset

        # Scale and quantize the membrane potential
        new_v_scaled = quantize_with_V(new_v)

        #print(new_v, new_v_scaled)
        # Determine spiking output
        
        v_scaled = (new_v_scaled - self.thr)/self.thr
        new_z    =  SpikeFunction(v_scaled, self.dampening_factor)

        new_state = (new_v_scaled, new_z)

        
        return new_z, new_state