
# Visualize gradient descent on a quadratic function
from matplotlib import pyplot as plt
import numpy as np

def quadratic(x):
    return x**2

# Gradient of the quadratic function
def grad_quadratic(x):
    return 2 * x

# Gradient descent parameters
x = 3.0  # initial position
learning_rate = 0.1 # size of the step taken
steps = 20

x_vals = [x]
y_vals = [quadratic(x)]

# Perform gradient descent
for _ in range(steps):
    grad = grad_quadratic(x)
    x = x - learning_rate * grad # take a small step in the opposite direction of gradient
    x_vals.append(x)
    y_vals.append(quadratic(x))

# Plot the function and the ball's path
x_plot = np.linspace(-3.5, 3.5, 200)
y_plot = quadratic(x_plot)

plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_plot, label='Quadratic Function: $y = x^2$')
plt.scatter(x_vals, y_vals, color='red', zorder=5, label='Ball Position')
plt.plot(x_vals, y_vals, color='red', linestyle='--', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Visualization')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Gradient descent on quadratic function using PyTorch SGD optimizer
x_torch = torch.tensor(3.0, requires_grad=True) 
optimizer = torch.optim.SGD([x_torch], lr=learning_rate)
steps = 20

x_vals_torch = [x_torch.item()]
y_vals_torch = [x_torch.item() ** 2]

for _ in range(steps):
    y = x_torch ** 2
    y.backward()
    optimizer.step()
    optimizer.zero_grad()
    x_vals_torch.append(x_torch.item())
    y_vals_torch.append(x_torch.item() ** 2)

plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_plot, label='Quadratic Function: $y = x^2$')
plt.scatter(x_vals_torch, y_vals_torch, color='green', zorder=5, label='PyTorch Ball Position (SGD)')
plt.plot(x_vals_torch, y_vals_torch, color='green', linestyle='--', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Visualization (PyTorch SGD)')
plt.legend()
plt.grid(True)
plt.show()