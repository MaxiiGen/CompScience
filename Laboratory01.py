import numpy as np
import matplotlib.pyplot as plt

# Example dataset
x = np.array([1, 2, 3, 4, 5])  # Independent variable
y = np.array([2, 3.5, 4.2, 5.8, 6.1])  # Dependent variable

# Calculate means of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate slope (m) and y-intercept (b)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
m = numerator / denominator
b = y_mean - m * x_mean

# Predicted values
y_pred = m * x + b

# Plot the data and regression line
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, y_pred, color="red", label=f"y = {m:.2f}x + {b:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Manual Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# Output slope and intercept
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")