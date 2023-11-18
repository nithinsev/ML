import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)
df_numerical = pd.get_dummies(df.iloc[:, :-1])
X = df_numerical 
y = df['PlayTennis']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)
tree_rules = export_text(model, feature_names=list(X.columns))
print("Decision Tree Rules:\n", tree_rules)
new_sample = pd.DataFrame({
    'Outlook_Sunny': [0],  
    'Outlook_Overcast': [1],
    'Outlook_Rain': [0],
    'Temperature_Cool': [1],
    'Temperature_Hot': [0],
    'Temperature_Mild': [0],
    'Humidity_High': [1],
    'Humidity_Normal': [0],
    'Windy_False': [0],
    'Windy_True': [1],
})

prediction = model.predict(new_sample)
print("Prediction for the new sample:", prediction[0])
