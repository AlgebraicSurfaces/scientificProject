# Hello

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from apache_beam.typehints.opcodes import unary

from model import SimpleDense, SimpleModel

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()


my_model = SimpleModel(units = 1, activation=None)

def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


learning_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = my_model.predict(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [my_model.layer.W, my_model.layer.b])
    my_model.layer.W.assign_sub(grad_loss_wrt_W * learning_rate)
    my_model.layer.b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss


for step in range(40):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

predictions = my_model.predict(inputs)

x = np.linspace(-1, 4, 100)
y = - my_model.layer.W[0] / my_model.layer.W[1] * x + (0.5 - my_model.layer.b) / my_model.layer.W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
