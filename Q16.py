import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
svm_classifier = SVC(random_state=42)
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifiers = {
    'Decision Tree': decision_tree_classifier,
    'SVM': svm_classifier,
    'Random Forest': random_forest_classifier
}
for name, classifier in classifiers.items():
    print(f"\n{name} Classifier:")
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
