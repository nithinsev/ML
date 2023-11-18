import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
np.random.seed(42)
data_size = 1000
X = np.random.randint(20000, 100000, size=(data_size, 1)) 
X = np.concatenate([X, np.random.randint(300, 850, size=(data_size, 1))], axis=1) 
X = np.concatenate([X, np.random.randint(5000, 50000, size=(data_size, 1))], axis=1) 
y = np.random.choice([0, 1], size=(data_size,))
columns = ['Income', 'Credit_Score', 'Loan_Amount', 'Loan_Approval']
df = pd.DataFrame(np.column_stack([X, y]), columns=columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)
predictions = naive_bayes_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
