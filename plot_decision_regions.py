import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
def plot_decision_regions(x, y, classifier, resolution = 0.02):
    print(y)
    markers = ('o', 's', '^', 'v', '<')
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    # x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x_min, x_max = x.min(axis = 0) - 1, x.max(axis = 0) + 1
    xx1, xx2 = np.meshgrid((np.arange(x_min[0], x_max[0], resolution), np.arange(x_min[1], x_max[1]), resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    plt.contourf(xx1, xx2, lab, alpha = 0.3, cmap = cmap)