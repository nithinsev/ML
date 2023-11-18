import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
np.random.seed(42)
data1 = np.random.normal(0, 1, 300).reshape(-1, 1)
data2 = np.random.normal(5, 1, 300).reshape(-1, 1)
data = np.vstack((data1, data2))
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data)
predictions = gmm.predict(data)
plt.scatter(data, np.zeros_like(data), c=predictions, cmap='viridis', s=50, edgecolors='k')
plt.title('Expectation-Maximization (EM) Algorithm - Gaussian Mixture Model (GMM)')
plt.xlabel('Data Points')
plt.yticks([])
plt.show()
