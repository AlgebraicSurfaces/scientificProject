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

my_model.fit(inputs, targets, epoch = 40, learning_rate=0.1)

predictions = my_model.predict(inputs)

x = np.linspace(-1, 4, 100)
y = - my_model.layer.W[0] / my_model.layer.W[1] * x + (0.5 - my_model.layer.b) / my_model.layer.W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
