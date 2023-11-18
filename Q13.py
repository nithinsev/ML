import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
np.random.seed(42)
data_size = 100
X = np.random.randint(2000, 2023, size=(data_size, 1)) 
X = np.concatenate([X, np.random.randint(10000, 100000, size=(data_size, 1))], axis=1)  
X = np.concatenate([X, np.random.uniform(1.0, 5.0, size=(data_size, 1))], axis=1) 
y = 15000 + 2000 * X[:, 0] - 0.1 * X[:, 1] + 5000 * X[:, 2] + np.random.randn(data_size) * 5000
columns = ['Year', 'Mileage', 'Engine_Size', 'Price']
df = pd.DataFrame(np.column_stack([X, y]), columns=columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)
predictions = linear_regression_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
plt.scatter(X_test[:, 0], y_test, color='black', label='Actual Prices')
plt.scatter(X_test[:, 0], predictions, color='red', label='Predictions')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.title('Car Price Prediction')
plt.show()
