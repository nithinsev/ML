import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
np.random.seed(42)
data_size = 1000
X = np.random.randint(1000, 3000, size=(data_size, 1)) 
X = np.concatenate([X, np.random.randint(1, 6, size=(data_size, 1))], axis=1) 
X = np.concatenate([X, np.random.randint(1, 4, size=(data_size, 1))], axis=1)  
y = 50000 + 100 * X[:, 0] + 20000 * X[:, 1] + 30000 * X[:, 2] + np.random.randn(data_size) * 5000
columns = ['Size', 'Bedrooms', 'Bathrooms', 'Price']
df = pd.DataFrame(np.column_stack([X, y]), columns=columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decision_tree_regressor = DecisionTreeRegressor(random_state=42)
decision_tree_regressor.fit(X_train, y_train)
predictions = decision_tree_regressor.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
plt.scatter(X_test[:, 0], y_test, color='black', label='Actual Prices')
plt.scatter(X_test[:, 0], predictions, color='red', label='Predictions')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.title('House Price Prediction')
plt.show()
