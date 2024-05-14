import numpy as np
import pandas as pd


def add_constant(x: np.ndarray):
    return np.append(np.ones((len(x), 1)), x, axis=1)


def least_squares(X: pd.DataFrame, y):
    """
    Return the best fit line for the given data.
    It works by finding the slope of the line of best fit and then using the slope to find the y-intercept.

    DISCLAIMER: I know it's very sensitive to outliers but it's so much faster than gradient descent.
    """
    return np.array(
        [
            np.mean(y)
            - np.mean(X, axis=0).dot(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)),
            *np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y),
        ]
    )