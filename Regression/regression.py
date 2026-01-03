print("Júlia Gualdi Schlichting")

import numpy as np
import matplotlib.pyplot as plt

def f_true (x):
    return 2 + 0.8*x


xs = np.linspace(-3,3,100)
ys = np.array([f_true(x) + np.random.randn()*0.5 for x in xs])

def h(x, theta):
    return theta[0] + theta[1]*x

def J(theta, xs, ys):
    m = len(xs)
    return (1/(2*m)) * np.sum((h(xs, theta) - ys)**2)

def gradient( theta, xs, ys):
    m = len(xs)
    error = h(xs, theta) - ys
    dtheta0 = (1/m) * np.sum(error)
    dtheta1 = (1/m) * np.sum(error * xs)
    return np.array([dtheta0, dtheta1])

def print_model(theta, xs, ys):
    plt.scatter(xs, ys, label="Data")
    plt.plot(xs, f_true(xs), "g-", label="f_true (original)")
    plt.plot(xs, h(xs, theta), "r-", label=f"h(x) = {theta[0]:.2f} + {theta[1]:.2f}x")
    plt.xlabel("Input x")
    plt.ylabel("Target y")
    plt.legend()
    plt.show()


theta = np.array([0.0, 0.0])  
alpha = 0.05  
num_iter = 200

for i in range(num_iter):
    grad = gradient(theta, xs, ys)
    theta -= alpha * grad
    if i % 50 == 0:
        print(f"Iteration {i}, J = {J(theta, xs, ys):.4f}, θ = {theta}")
        print_model(theta, xs, ys)

print("Final result:", theta)
print_model(theta, xs, ys)

# 2: Cost function for different learning rates 
def train(theta_init, xs, ys, alpha, num_iter):
    theta = theta_init.copy()
    cost_history = []
    for i in range(num_iter):
        grad = gradient(theta, xs, ys)
        theta -= alpha * grad
        cost_history.append(J(theta, xs, ys))
    return cost_history

num_iter_2 = 5000
alphas = [0.9, 0.1, 0.0001]
theta_init = np.array([0.0, 0.0])

for alpha in alphas:
    cost_history = train(theta_init, xs, ys, alpha, num_iter_2)
    plt.figure(figsize=(7,5))
    plt.plot(cost_history, label=f"alpha = {alpha}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.title(f"Cost function over iterations for alpha = {alpha}")
    plt.legend()
    plt.show()

theta0_vals = np.linspace(-1, 5, 100)
theta1_vals = np.linspace(-1, 2, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        J_vals[i,j] = J(np.array([t0, t1]), xs, ys)

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

plt.figure(figsize=(7,5))
cp = plt.contourf(T0, T1, J_vals.T, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.xlabel("theta0")
plt.ylabel("theta1")
plt.title("Cost function J(theta0, theta1)")
plt.show()