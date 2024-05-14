import numpy as np
from .util.least_squares import least_squares
class UnfittedModelException(Exception):
    """Exception raised when prediction is attempted before fitting."""

    def __init__(self):
        super().__init__('Model is not fitted yet. Call fit() method first.')


class LinearRegression:
    """Linear regression model."""
    weights: list[float | int] | None = None

    def __init__(self, fitting_method=least_squares):
        return None

    @staticmethod
    def __predict(x, weights):
        if isinstance(x, (int, float)):
            return weights[1] * x + weights[0]
        return weights[0] + np.dot(x, weights[1:])

    def _pre_fit(self, x, y):
        if len(x) != len(y):
            raise ValueError("Length of x and y must be the same.")

        def __type(x):
            return [x] if isinstance(x, (int, float)) else x

        return __type(x), __type(y)

    def fit(self, x, y):
        x, y = self._pre_fit(x, y)
        self.weights = fitting_method(x, y)

    def predict(self, x):
        if self.weights is None:
            raise UnfittedModelException()
        return self.__predict(x, self.weights)
    
    def __repr__(self) -> str:
        return f"LinearRegression(b={self.weights[0]}, W={self.weights[1:]})"


class GDLinearRegression(LinearRegression):
    rate: float | int = 0.05
    threshold: float | int = 1e-15

    def __init__(self, learning_rate=0.05, threshold=1e-15):
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        if threshold <= 0:
            raise ValueError("Threshold must be greater than 0.")
        self.rate = learning_rate
        self.threshold = threshold

    def fit(self, x, y):
        x, y = self._pre_fit(x, y)
        self.weights = gd(x, y, self.rate, self.threshold)
