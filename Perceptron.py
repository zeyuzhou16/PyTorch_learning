import numpy as np
class Perceptron:
    def __init__(self, eta = 0.1, random_state = 1, n_iter = 50)->None:      
        self.eta = eta
        self.random_state = random_state
        self.max_iter = n_iter
    def fit(self, X_train, y_train) -> None:
        # Note that X_train has row # = sample size and column # = features size
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(size = len(X_train[0]))
        self.b_ = np.float32(0.0)
        self.errors_ = []
        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X_train, y_train):
                update = self.eta * (yi - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
            self.errors_.append(errors)
        return 
    def predict(self, X_test) -> np.ndarray:
        self.y_temp = X_test.dot(self.w_) + self.b_
        return np.where(self.y_temp >= 0.0, 1, 0)
    def scoring(self, X_test, y_test) -> np.float32:
        return (self.predict(X_test) == y_test).sum() / np.float32(len(y_test))
        

# class Perceptron:
#     """Perceptron classifier.

#     Parameters
#     ------------
#     eta : float
#       Learning rate (between 0.0 and 1.0)
#     n_iter : int
#       Passes over the training dataset.
#     random_state : int
#       Random number generator seed for random weight
#       initialization.

#     Attributes
#     -----------
#     w_ : 1d-array
#       Weights after fitting.
#     b_ : Scalar
#       Bias unit after fitting.
#     errors_ : list
#       Number of misclassifications (updates) in each epoch.

#     """
#     def __init__(self, eta=0.01, n_iter=50, random_state=1):
#         self.eta = eta
#         self.n_iter = n_iter
#         self.random_state = random_state

#     def fit(self, X, y):
#         """Fit training data.

#         Parameters
#         ----------
#         X : {array-like}, shape = [n_examples, n_features]
#           Training vectors, where n_examples is the number of examples and
#           n_features is the number of features.
#         y : array-like, shape = [n_examples]
#           Target values.

#         Returns
#         -------
#         self : object

#         """
#         rgen = np.random.RandomState(self.random_state)
#         self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
#         self.b_ = np.float_(0.)
        
#         self.errors_ = []

#         for _ in range(self.n_iter):
#             errors = 0
#             for xi, target in zip(X, y):
#                 update = self.eta * (target - self.predict(xi))
#                 self.w_ += update * xi
#                 self.b_ += update
#                 errors += int(update != 0.0)
#             self.errors_.append(errors)
#         return self

#     def net_input(self, X):
#         """Calculate net input"""
#         return np.dot(X, self.w_) + self.b_

#     def predict(self, X):
#         """Return class label after unit step"""
#         return np.where(self.net_input(X) >= 0.0, 1, 0)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.arange(100)
    X = np.zeros((100, 2))
    X[:, 0] = x
    X[:, 1] = x ** 2 + 2
    print(x[0].dtype)
    y = np.concatenate((np.zeros(50), np.ones(50)), axis = 0)
    ppn = Perceptron(eta = 0.01, n_iter = 10)
    ppn.fit(X_train = X, y_train= y)
    plt.figure(1)
    plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'Setosa')
    plt.scatter(X[50:, 0], X[50:, 1], color = 'blue', marker = 's', label = 'Versicolor')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.figure(2)
    plt.plot(np.arange(len(ppn.errors_)) + 1, ppn.errors_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()