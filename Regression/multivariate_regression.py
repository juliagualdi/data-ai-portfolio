print("Júlia Gualdi Schlichting")

import numpy as np
import matplotlib.pyplot as plt

def f_true(x1, x2):
    return 2 + 0.8*x1 + 1.5*x2

np.random.seed(42)

x1s = np.linspace(-3, 3, 100)
x2s = np.random.uniform(-3, 3, 100)
ys = np.array([f_true(x1, x2) + np.random.randn()*0.5 for x1, x2 in zip(x1s, x2s)])

def h(X, theta):
    return np.dot(X, theta)

def J(theta, X, ys):
    m = len(ys)
    return (1/(2*m)) * np.sum((h(X, theta) - ys)**2)

def gradient(theta, X, ys):
    m = len(ys)
    error = h(X, theta) - ys
    return (1/m) * X.T.dot(error)

def print_model(theta, x1s, x2s, ys):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1s, x2s, ys, label="Data")
    X_plot = np.c_[np.ones(len(x1s)), x1s, x2s]
    ax.scatter(x1s, x2s, h(X_plot, theta), color="r", label="h(x)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    plt.legend()
    plt.show()

X = np.c_[np.ones(len(x1s)), x1s, x2s]
theta = np.zeros(3)
alpha = 0.05
num_iter = 500

for i in range(num_iter):
    grad = gradient(theta, X, ys)
    theta -= alpha * grad
    if i % 100 == 0:
        print(f"Iteration {i}, J = {J(theta, X, ys):.4f}, θ = {theta}")

print("Final result:", theta)
print_model(theta, x1s, x2s, ys)

# 2: Cost function for different learning rates 

def train(theta_init, X, ys, alpha, num_iter):
    theta = theta_init.copy()
    cost_history = []
    for i in range(num_iter):
        grad = gradient(theta, X, ys)
        theta -= alpha * grad
        cost_history.append(J(theta, X, ys))
    return cost_history

num_iter_2 = 2000
alphas = [0.5, 0.1, 0.01]
theta_init = np.zeros(3)

for alpha in alphas:
    cost_history = train(theta_init, X, ys, alpha, num_iter_2)
    plt.figure(figsize=(7,5))
    plt.plot(cost_history, label=f"alpha = {alpha}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.title(f"Cost function over iterations for alpha = {alpha}")
    plt.legend()
    plt.show()
