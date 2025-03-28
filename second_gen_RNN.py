import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Activation,  Dropout,LSTM,Embedding,Layer, LSTM, Conv1D, MaxPool1D, TimeDistributed, Input, GRU, RNN
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow.keras as keras

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


optimizer     = keras.optimizers.Adam(0.001)

def my_loss(y_true,y_pred):
    
    squared_difference = tf.square(y_true - y_pred)
    
    return tf.reduce_mean(squared_difference, axis=[1, 0])


def model_klassiche():
    inputs = Input((None, 12))
    
    cell1 = tf.keras.layers.SimpleRNNCell(units=208)
    cell2 = tf.keras.layers.SimpleRNNCell(units=208)
    cell3 = tf.keras.layers.SimpleRNNCell(units=208)
    cell4 = tf.keras.layers.SimpleRNNCell(units=208)
    
    
    
    layer_1     = RNN(cell1, return_sequences=True)
    
    layer_2     = RNN(cell2, return_sequences=True)
    
    layer_3     = RNN(cell3, return_sequences=True)
    
    layer_4     = RNN(cell4, return_sequences=True)
    
    X           = layer_1(inputs)
    
    X           = layer_2(X)
    
    X           = layer_3(X)
    
    V           = layer_4(X)
    
    
    V  = Dense(240, activation='tanh')(V)
    
    V  = Dense(64, activation='tanh')(V)
    
    V1  = Dense(1, activation='tanh')(V)
    
    
    model_RNN = Model(inputs=inputs, outputs=[V1])
    
    
    opt = Adam(0.001)
    
    model_RNN.compile(optimizer = opt, loss = my_loss)

    return model_RNN

class TrainModel_RNN(keras.Model):
    
    def __init__(self):
        
        super().__init__()
        

        cell1 = tf.keras.layers.SimpleRNNCell(units=208)
        cell2 = tf.keras.layers.SimpleRNNCell(units=208)
        cell3 = tf.keras.layers.SimpleRNNCell(units=208)
        cell4 = tf.keras.layers.SimpleRNNCell(units=208)
        

        

        self.hidden1               = tf.keras.layers.RNN(cell1, return_sequences=True, unroll=True)
        self.hidden2               = tf.keras.layers.RNN(cell2, return_sequences=True, unroll=True)
        self.hidden3               = tf.keras.layers.RNN(cell3, return_sequences=True, unroll=True)        
        
        

        self.Deco                  = tf.keras.layers.RNN(cell4, return_sequences=True, unroll=True)


        self.d1                            = Dense(240, activation='tanh')
        self.d2                            = Dense(64, activation='tanh')
        self.d3                            = Dense(1, activation='linear')
        
        
    
        
    def call(self, x):

       
        
        x = self.hidden1(x)

        x = self.hidden2(x)
        
        x = self.hidden3(x)

        #x = self.hidden4(x)

        # x = self.hidden5(x)

        #x = self.out(x)
        
        x = self.Deco(x)

        x = tf.math.tanh(x)

        #x = self.Norm(x)

        
        x = self.d1(x)
        
        x = self.d2(x)
        
        x = self.d3(x)
        
        
        return x
    

def copy_model_RNN(model, x):
    '''Copy model weights to a new model.
    
    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    copied_model = TrainModel_RNN()
    
    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model(tf.convert_to_tensor(x))
    
    copied_model.set_weights(model.get_weights())
    
    return copied_model




def compute_loss(model, x, y, loss_fn=my_loss):
    logits = model(x)
    mse = loss_fn(y, logits)
    return mse, logits


def MAML_LMU_multiple_adapt_inner_outer(model, train_dataset, test_dataset, input_nn_test, output_nn_test, input_nn_validation_inp, input_nn_validation_output, epochs, inner_epochs):

    lr_inner = 0.001
    val_loss = [0]
    for tt in range(epochs):

        total_loss = 0
        losses = []
        
        meta_grads = [tf.zeros_like(w) for w in model.trainable_variables]
        # Step 3 and 4

        
        weights_outer   =  model.get_weights()
        for i, (x, y) in enumerate(train_dataset):
    
            
            with tf.GradientTape() as test_tape:
                # test_tape.watch(model.trainable_variables)
                # Step 5
                #weights = model.get_weights()
                
                for ee in range(inner_epochs):

                    adapted_weights = []
                    with tf.GradientTape(persistent=True) as train_tape:
                        train_loss, _ = compute_loss(model, x, y)
    
                    # Step 6
                    #print(f'Train loss: {train_loss}')
                    gradients = train_tape.gradient(train_loss, model.trainable_variables)
                    k = 0
                    
                    model_copy = copy_model_RNN(model, x)
    
                    for j in range(len(model_copy.layers)):
                        

                        if isinstance(model_copy.layers[j], tf.keras.layers.RNN):
                            
                            model_copy.layers[j].cell.kernel = tf.subtract(model.layers[j].cell.kernel,
                                        tf.multiply(lr_inner, gradients[k]))
                            model_copy.layers[j].cell.recurrent_kernel = tf.subtract(model.layers[j].cell.recurrent_kernel,
                                    tf.multiply(lr_inner, gradients[k+1]))
                            model_copy.layers[j].cell.bias = tf.subtract(model.layers[j].cell.bias,
                                        tf.multiply(lr_inner, gradients[k+2]))

                            adapted_weights.append(tf.subtract(model.layers[j].cell.kernel,
                                        tf.multiply(lr_inner, gradients[k])))
                            adapted_weights.append(tf.subtract(model.layers[j].cell.recurrent_kernel,
                                    tf.multiply(lr_inner, gradients[k+1])))
                            adapted_weights.append(tf.subtract(model.layers[j].cell.bias,
                                        tf.multiply(lr_inner, gradients[k+2])))
                            
                            k += 3
                            

                        elif isinstance(model_copy.layers[j], tf.keras.layers.Dense):
                            
                            #print(j+1, 'Dense layer')
                            
                            model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                    tf.multiply(lr_inner, gradients[k]))
                            model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
                                    tf.multiply(lr_inner, gradients[k+1]))

                            adapted_weights.append(tf.subtract(model.layers[j].kernel,
                                    tf.multiply(lr_inner, gradients[k])))
                            adapted_weights.append(tf.subtract(model.layers[j].bias,
                                    tf.multiply(lr_inner, gradients[k+1])))
                            k += 2


                    
                    model.set_weights(adapted_weights)
                
                # Step 8
                test_loss, logits = compute_loss(model_copy, input_nn_test, output_nn_test)

            model.set_weights(weights_outer)
            # Step 8
            gradients_ = test_tape.gradient(test_loss, model.trainable_variables)
            
            meta_grads = [g + mg for g, mg in zip(gradients_, meta_grads)]
            
            
        
            #print('Hi')
            # Logs
            total_loss += test_loss
        loss = total_loss / (i+1.0)
        losses.append(loss)

    #         if loss.numpy()<1e-2:
    #             break
        print('Loss: '+str(np.linalg.norm(loss)), 'Epoch nr: '+str(tt+1))
        optimizer.apply_gradients(zip(meta_grads, model.trainable_variables))
        meta_gradient_norm = tf.norm(tf.concat([tf.reshape(g, [-1]) for g in meta_grads if g is not None], axis=0))
        
        model_new = model_klassiche()
        updated_weights =  model.get_weights()
        model_new.set_weights(updated_weights)
        erst_l = model_new.evaluate(input_nn_validation_inp, input_nn_validation_output)
        val_loss.append(erst_l)
        if meta_gradient_norm<1e-4 or val_loss[-1]<1e-4:
            break
                #start = time.time()
                
    return loss, losses




