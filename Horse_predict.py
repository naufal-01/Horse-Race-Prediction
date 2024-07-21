import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load datasets
races = pd.read_csv('/content/drive/MyDrive/horse riding/races_2020.csv')
horses = pd.read_csv('/content/drive/MyDrive/horse riding/horses_2020.csv')
forward = pd.read_csv('/content/drive/MyDrive/horse riding/forward.csv')

# Merge datasets
# Merge races and horses on 'rid'
merged_data_rh = pd.merge(races, horses, on='rid', how='inner')
print(f"Shape after merging races and horses: {merged_data_rh.shape}")

# Merge with forward dataset on common columns (adjust if necessary)
merged_data = pd.merge(merged_data_rh, forward, on=['course'], how='inner')
print(f"Shape after merging with forward: {merged_data.shape}")

# Check if columns needed for further processing are present
print("Columns in merged_data:", merged_data.columns)

# Fill missing values
merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')

# Identify categorical columns
categorical_cols = merged_data.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    merged_data[col] = le.fit_transform(merged_data[col].astype(str))

# Ensure 'outcome' column exists and is properly assigned
# For demonstration, we'll create a dummy 'outcome' column
# Replace this with actual logic to define the 'outcome' if you have it
if 'outcome' not in merged_data.columns:
    merged_data['outcome'] = (merged_data['position'] == 1).astype(int)  # Example: Win if position == 1

# Split data into features and target
X = merged_data.drop(['outcome'], axis=1)
y = merged_data['outcome']

# Check if there are samples available
if X.shape[0] > 0 and y.shape[0] > 0:
    # Handle imbalanced data
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
else:
    raise ValueError("No samples available after merging. Check merge keys and dataset integrity.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
models = {
    'RandomForest': RandomForestClassifier(),
    # Add other models as needed
}

# Cross-validation to assess model performance
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f'{name} CV Accuracy: {np.mean(cv_scores)}')

# Feature Selection
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train_rfe, y_train)

# Best model
best_model = grid_search.best_estimator_

# Model evaluation
y_pred = best_model.predict(X_test_rfe)
print('Classification Report:', classification_report(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
