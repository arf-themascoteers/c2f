import matplotlib.pyplot as plt
import numpy as np


def celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32


SAMPLE_SIZE = 100
LEARNING_RATE = 0.0001

celsius_values = np.linspace(-50, 50, 100)
celsius_values = celsius_values.reshape(SAMPLE_SIZE, 1)
fahrenheit_values = celsius_to_fahrenheit(celsius_values)

weights1 = np.random.randn(1, 1)
biases1 = np.random.randn(1, 2)

weights2 = np.random.randn(2, 1)
biases2 = np.random.randn(1, 1)


def forward():
    output1 = np.dot(celsius_values, weights1) + biases1
    output2 = np.dot(output1, weights2) + biases2
    loss = np.square(output2 - fahrenheit_values).sum() / SAMPLE_SIZE
    return output1, output2, loss


def backward(output1, output2):
    grad_output2 = 2 * (output2 - fahrenheit_values) / SAMPLE_SIZE
    grad_weights_2 = np.dot(output1.T, grad_output2)
    grad_biases_2 = np.sum(grad_output2)

    grad_output1 = np.dot(grad_output2, weights2.T)
    grad_weights_1 = np.dot(celsius_values.T, grad_output1)
    grad_biases_1 = np.sum(grad_output1)

    return grad_weights_1, grad_biases_1, grad_weights_2, grad_biases_2


def update_parameters(grad_weights_1, grad_biases_1, grad_weights_2, grad_biases_2):
    global weights2, biases2, weights1, biases1
    weights2 = weights2 - (LEARNING_RATE * grad_weights_2)
    biases2 = biases2 - (LEARNING_RATE * grad_biases_2)

    weights1 = weights1 - (LEARNING_RATE * grad_weights_1)
    biases1 = biases1 - (LEARNING_RATE * grad_biases_1)


def train():
    for t in range(2000):
        output1, output2, loss = forward()
        grad_weights_1, grad_biases_1, grad_weights_2, grad_biases_2 = backward(output1, output2)
        update_parameters(grad_weights_1, grad_biases_1, grad_weights_2, grad_biases_2)


def get_predicted_fahrenheit_values():
    output1, prediction, loss = forward()
    return prediction


predicted_before_training = get_predicted_fahrenheit_values()
train()
predicted_after_training = get_predicted_fahrenheit_values()
fig, ax = plt.subplots()
ax.plot(celsius_values, fahrenheit_values, '-b', label='Actual')
ax.plot(celsius_values, predicted_before_training, '--r', label='Predicted before Training')
ax.plot(celsius_values, predicted_after_training, '--g', label='Predicted after Training')
plt.xlabel("Celsius")
plt.ylabel("Fahrenheit")
ax.legend(frameon=False)
plt.show()
