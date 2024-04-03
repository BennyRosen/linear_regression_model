import numpy as np
import matplotlib.pyplot as plt

# Utilizing the advanced features of numpy
class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10):
        # Learning rate is often denoted mathematically as the greek letter alpha (a)
        # self.X will be an m * n matrix
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.X = None
        self.y_targets = None
        self.weights = None
        self.b = None

    def fit(self, X, y_targets, weights=None, b=None):
        print("Running fit...")
        self.X = np.array(X)
        self.y_targets = np.array(y_targets)
        self.weights = self.rand_weights(self.X.shape) if weights is None else weights
        self.b = self.rand_intercept() if b is None else b
    
        print(self.mse())
        for i in range(self.num_iterations):
            self.gradient_descent()
            self.eval(i + 1)
    
    def hypothesis(self):
        return np.dot(self.X, self.weights) + self.b # y = b + (w1*x1) + (w2+x2 ).. + (wn * xn )
    
    def predict(self, x):
        return np.dot(x, self.weights) + self.b
    
    # Cost functions show the models error, "How BAD the model is at predicting"
    def mse(self):
        predictions = self.hypothesis()
        squaredError = (predictions - self.y_targets) ** 2
        squaredErrorSum = np.sum(squaredError)
        mse = squaredErrorSum / (2 * len(self.y_targets))
        return mse
            
    # Optimization functions tune the weights in our models, to better fit the data
    def gradient_descent(self):
        predictions = self.hypothesis() 
        error = predictions - self.y_targets
        self.weights -= self.learning_rate * np.dot(self.X.T, error)
        self.b -= self.learning_rate * np.mean(error)
        
    # np.random.rand() selects a random number between the interval [0, 1]
    def rand_weights(self, shape):
        weights = []
        for i in range(shape[1]):
            weights.append(np.random.rand())
        return np.array(weights)
    
    def rand_intercept(self):
        return np.random.rand()
    
    def eval(self, num):
        print("-------------")
        print(f"MSE for iteration #{num} = {self.mse()}")

    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, X):
        self._X = np.array(X)

    @property
    def y_targets(self):
        return self._y_targets

    @y_targets.setter
    def y_targets(self, y_targets):
        self._y_targets = np.array(y_targets)

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = np.array(weights)

    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, b):
        self._b = np.array(b)

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
    
    @property
    def num_iterations(self):
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, num_iterations):
        self._num_iterations = num_iterations



# Method implementations without the efficient use of numpy
    
# def predict(self):
#     predictions = []
#     for i in range(len(self.X)):
#         _sum = 0
#         for j in range(len(self.X[i])):
#             _sum += self.X[i][j] * self.weights[j]
#         predictions.append(_sum + self.b[i])

#     return predictions 

# def mse(self):
#     predictions = self.predict()

#     _sum = 0
#     for i in range(len(predictions)):
#         _sum += (predictions[i] - self.y_targets[i]) ** 2

#     return _sum / 2(self.y_targets)