import tensorflow as tf
from tensorflow.keras.layers import Layer



class RMSNorm(Layer):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = tf.ones(dim)


    def _norm(self,x):

         return x/tf.sqrt(tf.reduce_mean(tf.square(x),axis=-1,keepdims=True) + self.eps)
    

    def call(self,x):

        return tf.cast(self._norm(x)*self.weight, x.dtype)


        

