from .linear_regression import GDLinearRegression
from .util.gradient_descent import gradient_descent

class LogisticRegression(GDLinearRegression):
    """Logistic regression model."""

    def __init__(self, learning_rate=0.05, threshold=1e-6):
        super().__init__(learning_rate, threshold)
    
    @staticmethod    
    def __predict(x, weights):
        return 1 / (1 + np.exp(-(x.dot(weights[1:]) + weights[0])))
    
    @staticmethod
    def __pred(x, w):
        return 1 / (1 + np.exp(- (x.dot(w))))
    
    @staticmethod
    def __cost(y_pred, y):
        """Logistic Regression cost function"""
        return sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))/ -len(y)

    def fit(self, X, y):
        X, y = self._pre_fit(X, y)
        self.weights = gradient_descent.gradient_descent(X, y, self.rate, self.threshold, pred=self.__pred, cost_fn=self.__cost, verbose=True)

    def predict(self, X):
        return 1 if self.__predict(X, self.weights) > 0.5 else 0