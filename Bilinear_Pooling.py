from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf

class Bilinear_Pooling(Layer):

    def __init__(self,  **kwargs):
        super(Bilinear_Pooling, self).__init__(**kwargs)

    def build(self, input_shape):        
        super(Bilinear_Pooling, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, Inputs):
        Reshaped_Inputs = K.reshape(Inputs,[-1,Inputs.get_shape().as_list()[1]*Inputs.get_shape().as_list()[2],
                                            Inputs.get_shape().as_list()[3]])
        Bilinear_Pooling = K.batch_dot(Reshaped_Inputs,Reshaped_Inputs,axes = [1,1])        
        Signed_Sqrt = K.sign(Bilinear_Pooling) * K.sqrt(K.abs(Bilinear_Pooling)+1e-9)
        return  K.l2_normalize(Signed_Sqrt, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[3],input_shape[3])