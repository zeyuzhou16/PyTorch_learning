import numpy as np
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')
class AdalineGD2:
    def __init__(self, eta = 0.01, random_state = 1, n_iter = 50)->None:      
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
            errors = y_train - self.net_input(X_train)
            self.w_ += self.eta * 2.0 * X_train.T.dot(errors) / X_train.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            # for xi, yi in zip(X_train, y_train):
            #     update = self.eta * (yi - self.predict(xi))
            #     self.w_ += update * xi
            #     self.b_ += update
            self.errors_.append((errors ** 2).mean())
        return
    def net_input(self, X_test):
        self.y_temp = np.dot(X_test, self.w_) + self.b_
        return self.y_temp
    def predict(self, X_test) -> np.ndarray:
        self.net_input(X_test = X_test)
        return np.where(self.y_temp >= 0.50, 1, 0)
    def scoring(self, X_test, y_test) -> np.float32:
        return (self.predict(X_test) == y_test).sum() / np.float32(len(y_test))
class AdalineSGD2:
    def __init__(self, eta = 0.01, random_state = 1, n_iter = 50)->None:      
        self.eta = eta
        self.random_state = random_state
        self.max_iter = n_iter
    def fit(self, X_train, y_train) -> None:
        # Note that X_train has row # = sample size and column # = features size
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(size = X_train.shape[1])
        self.b_ = np.float32(0.0)
        self.errors_ = []
        for _ in range(self.max_iter):
            errors = y_train - self.net_input(X_train)
            self.w_ += self.eta * 2.0 * X_train.T.dot(errors).sum()
            self.b_ += self.eta * 2.0 * errors.sum()
            # for xi, yi in zip(X_train, y_train):
            #     update = self.eta * (yi - self.predict(xi))
            #     self.w_ += update * xi
            #     self.b_ += update
            self.errors_.append((errors ** 2).sum())
        return
    def net_input(self, X_test):
        self.y_temp = np.dot(X_test, self.w_) + self.b_
        return self.y_temp
    def predict(self, X_test) -> np.ndarray:
        self.net_input(X_test = X_test)
        return np.where(self.y_temp >= 0.50, 1, 0)
    def scoring(self, X_test, y_test) -> np.float32:
        return (self.predict(X_test) == y_test).sum() / np.float32(len(y_test))
class AdalineSGD:
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
      Mean squared error loss function value averaged over all
      training examples in each epoch.

        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.errors_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            # for xi, target in zip(X, y):
            #     losses.append(self._update_weights(xi, target))
            losses.append(self._update_weights(X, y))
            avg_loss = np.mean(losses)
            self.errors_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * np.dot(error, xi)
        self.b_ += self.eta * 2.0 * error.sum()
        loss = error**2
        return loss
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
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
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return 
    def predict(self, X_test) -> np.ndarray:
        self.y_temp = X_test.dot(self.w_) + self.b_
        return np.where(self.y_temp >= 0.0, 1, 0)
    def scoring(self, X_test, y_test) -> np.float32:
        return (self.predict(X_test) == y_test).sum() / np.float32(len(y_test))
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    from sklearn.decomposition import PCA
    import pandas as pd
    s = 'https://archive.ics.uci.edu/ml/'\
        'machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s, header = None, encoding = 'utf-8')
    print(df.head())
    X_temp = df.iloc[:, :3].to_numpy()
    pca = PCA(n_components=2)
    X = pca.fit_transform(X_temp)
    y = np.where(df.iloc[:, 4] == 'Iris-setosa', 0, 1)
    print(X)
    print(y)
    
    ppn = AdalineSGD(eta = 0.1, n_iter = 50)
    ppn.fit(X, y)
    plt.figure(1)
    plot_decision_regions(X = X, y = y, classifier=ppn)
    plt.scatter(X[y==0, 0], X[y==0, 1], color = 'red', marker = 'o', label = 'Setosa')
    plt.scatter(X[y==1, 0], X[y==1, 1], color = 'blue', marker = 's', label = 'Versicolor')
    
    plt.plot()
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.figure(2)
    plt.plot(np.arange(len(ppn.errors_)) + 1, ppn.errors_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()