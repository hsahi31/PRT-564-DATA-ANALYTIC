import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier

# Load the preprocessed dataset
data = pd.read_csv('retractions_cleaned.csv')

# Drop rows with missing values in crucial columns for classification
data.dropna(subset=['Subject_(B/T) Business - Accounting;(B/T) Business - Economics;', 'Institutions', 'Countries', 'ReasonCategory_Other'], inplace=True)

# Define the target variable
data['Retracted'] = 1  # Assuming all records in this dataset are retracted papers

# Encode categorical features
label_encoders = {}
for column in ['Subject_(B/T) Business - Accounting;(B/T) Business - Economics;', 'Institutions', 'Countries', 'ReasonCategory_Other']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data[['Subject_(B/T) Business - Accounting;(B/T) Business - Economics;', 'Institutions', 'Countries', 'ReasonCategory_Other']]
y = data['Retracted']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the target variable
data['Retracted'] = 1  # Assuming all records in this dataset are retracted papers

# Encode categorical features
label_encoders = {}
for column in ['Subject_(B/T) Business - Accounting;(B/T) Business - Economics;', 'Institutions', 'Countries', 'ReasonCategory_Other']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data[['Subject_(B/T) Business - Accounting;(B/T) Business - Economics;', 'Institutions', 'Countries', 'ReasonCategory_Other']]
y = data['Retracted']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Predict on the test set
y_pred_tree = tree.predict(X_test)

# Evaluate the model
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_precision = precision_score(y_test, y_pred_tree, average='weighted')
tree_recall = recall_score(y_test, y_pred_tree, average='weighted')
tree_f1 = f1_score(y_test, y_pred_tree, average='weighted')

print(f"Decision Tree - Accuracy: {tree_accuracy}, Precision: {tree_precision}, Recall: {tree_recall}, F1 Score: {tree_f1}")

# Initialize and train the Random Forest model
forest = RandomForestClassifier()
forest.fit(X_train, y_train)

# Predict on the test set
y_pred_forest = forest.predict(X_test)

# Evaluate the model
forest_accuracy = accuracy_score(y_test, y_pred_forest)
forest_precision = precision_score(y_test, y_pred_forest, average='weighted')
forest_recall = recall_score(y_test, y_pred_forest, average='weighted')
forest_f1 = f1_score(y_test, y_pred_forest, average='weighted')

print(f"Random Forest - Accuracy: {forest_accuracy}, Precision: {forest_precision}, Recall: {forest_recall}, F1 Score: {forest_f1}")