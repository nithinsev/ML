import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
degree = 2 
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)
polynomial_model = LinearRegression()
polynomial_model.fit(X_poly_train, y_train)
polynomial_predictions = polynomial_model.predict(X_poly_test)
linear_mse = mean_squared_error(y_test, linear_predictions)
polynomial_mse = mean_squared_error(y_test, polynomial_predictions)
plt.scatter(X, y, label='Actual Data')
plt.plot(X_test, linear_predictions, label=f'Linear Regression (MSE: {linear_mse:.2f})', color='red')
plt.plot(X_test, polynomial_predictions, label=f'Polynomial Regression (MSE: {polynomial_mse:.2f})', color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear vs Polynomial Regression')
plt.show()
