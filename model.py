import tensorflow as tf
from tensorflow import keras


class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros")

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


class SimpleModel:
    def __init__(self, units, activation=None):
        self.layer = SimpleDense(units, activation)

    def predict(self, inputs):

        y = self.layer(inputs)

        return y

    def training_step(self, inputs, targets, learning_rate = 0.1):
        with tf.GradientTape() as tape:
            predictions = self.predict(inputs)
            loss = square_loss(targets, predictions)
        grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [self.layer.W, self.layer.b])
        self.layer.W.assign_sub(grad_loss_wrt_W * learning_rate)
        self.layer.b.assign_sub(grad_loss_wrt_b * learning_rate)
        return loss

    def fit(self, inputs, targets, epoch = 40, learning_rate=0.1):
        for step in range(epoch):
            loss = self.training_step(inputs, targets, learning_rate)
            print(f"Loss at step {step}: {loss:.4f}")
        return loss

def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)