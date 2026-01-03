print("Júlia Gualdi Schlichting")

import numpy as np
import matplotlib.pyplot as plt

def f_true(x):
    return 2 + 0.8*x - 0.3*x**2 + 0.1*x**3

xs = np.linspace(-3, 3, 100)
ys = np.array([f_true(x) + np.random.randn()*0.5 for x in xs])

def h_poly(x, theta):
    return sum(theta[i] * x**i for i in range(len(theta)))

def J(theta, xs, ys):
    m = len(xs)
    return (1/(2*m)) * np.sum((h_poly(xs, theta) - ys)**2)

def gradient(theta, xs, ys):
    m = len(xs)
    error = h_poly(xs, theta) - ys
    grad = np.array([np.sum(error * xs**i)/m for i in range(len(theta))])
    return grad

def print_model(theta, xs, ys):
    plt.scatter(xs, ys, label="Data")
    plt.plot(xs, f_true(xs), "g-", label="f_true")
    plt.plot(xs, [h_poly(x, theta) for x in xs], "r-", label="h_poly")
    plt.xlabel("Input x")
    plt.ylabel("Target y")
    plt.legend()
    plt.show()

theta = np.zeros(4)
alpha = 0.01
num_iter = 5000

for i in range(num_iter):
    grad = gradient(theta, xs, ys)
    theta -= alpha * grad
    if i % 1000 == 0:
        print(f"Iter {i}, J = {J(theta, xs, ys):.4f}, θ = {theta}")

print("Final result:", theta)
print_model(theta, xs, ys)
