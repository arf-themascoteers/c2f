import matplotlib.pyplot as plt
import numpy as np

def celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32

SAMPLE_SIZE = 100
LEARNING_RATE = 0.01

celsius_values = np.linspace(-50, 50, 100).reshape(SAMPLE_SIZE,1)
fahrenheit_values = celsius_to_fahrenheit(celsius_values)

weight1 = np.random.randn(1,3)
bias1 = np.random.randn(1,3)

weight2 = np.random.randn(3,1)
bias2 = np.random.randn(1,1)

def forward(x,y):
    output1 = np.dot(x, weight1) + bias1
    output2 = np.dot(output1, weight2) + bias2
    loss = np.square(output2 - y).sum() / x.shape[0]
    return output1, output2, loss

def backward(x, y, output1, output2):
    grad_output2 = 2 * (output2 - y) / x.shape[0]
    grad_weight_2 = np.dot(output1.T, grad_output2) / y.shape[0]
    grad_bias_2 = np.sum(grad_output2, axis=0, keepdims=True).sum() / x.shape[0]

    grad_output1 = np.dot(grad_output2, weight2.T) / x.shape[0]
    grad_weight_1 = np.dot(x.T, grad_output1) / x.shape[0]
    grad_bias_1 = np.sum(grad_output1, axis=0, keepdims=True).sum() / x.shape[0]

    return grad_weight_1, grad_bias_1, grad_weight_2, grad_bias_2

for t in range(3000):
    output1, output2, loss = forward(celsius_values, fahrenheit_values)
    grad_weight_1, grad_bias_1, grad_weight_2, grad_bias_2 = backward(
        celsius_values, fahrenheit_values, output1, output2)

    weight2 = weight2 - (LEARNING_RATE * grad_weight_2)
    bias2 = bias2 - (LEARNING_RATE * grad_bias_2)

    weight1 = weight1 - (LEARNING_RATE * grad_weight_1)
    bias1 = bias1 - (LEARNING_RATE * grad_bias_1)

_,predicted,_ = forward(celsius_values, fahrenheit_values)
plt.plot(celsius_values, fahrenheit_values, 'b')
plt.plot(celsius_values, predicted, 'r+')
plt.show()