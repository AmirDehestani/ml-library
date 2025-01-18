import numpy as np
import matplotlib.pyplot as plt


def generate_noisy_sin_points(num_points, noise_level=0.1):
    """
    Generate noisy points that follow a sine function.

    Parameters:
        num_points (int): Number of points to generate.
        noise_level (float): Standard deviation of the noise.

    Returns:
        np.ndarray: x values.
        np.ndarray: y values.
    """
    x = np.linspace(0, 1, num_points)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, noise_level, num_points)
    return x, y


def polynomial_regression(x, y, degree):
    """
    Fit a polynomial regression model using the closed-form solution.
    
    Parameters:
        x (array-like): Independent variable values.
        y (array-like): Dependent variable values.
        degree (int): Degree of the polynomial.
    
    Returns:
        np.ndarray: Optimal weights (w*).
    """
    X = np.vander(x, degree + 1, increasing=True)
    w_star = np.linalg.pinv(X.T @ X) @ X.T @ y
    return w_star


def polynomial(x, w):
    """
    Evaluate a polynomial at x.

    Parameters:
        x (array-like): Independent variable values.
        w (array-like): Polynomial coefficients (weights).

    Returns:
        np.ndarray: Predicted values.
    """
    return sum(w[i] * x**i for i in range(len(w)))


def mean_squared_error(y, y_pred):
    """
    Calculate the mean squared error.

    Parameters:
        y (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Mean squared error.
    """
    return np.sum((y - y_pred)**2) / 2


if __name__ == "__main__":
    x, y = generate_noisy_sin_points(10)

    for i in [0, 1, 3, 9]:
        w_star = polynomial_regression(x, y, i)
        y_pred = polynomial(x, w_star)
        error_value = mean_squared_error(y, y_pred)
        print(f'Degree: {i}, Error: {error_value}')

        x_plot = np.linspace(0, 1, 100)
        y_plot = polynomial(x_plot, w_star)
        plt.plot(x_plot, y_plot, label=f'Degree {i}')

    plt.scatter(x, y, color='black')
    plt.plot(x_plot, np.sin(2 * np.pi * x_plot), label='sin(2πx)', linestyle='--')
    plt.legend()
    plt.show()