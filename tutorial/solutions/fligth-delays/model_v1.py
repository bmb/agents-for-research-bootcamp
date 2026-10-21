import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data/flights_clean.csv')

# Select features and target
features = ['Reporting_Airline', 'Origin', 'Dest', 'DayOfWeek', 'dep_hour', 'Distance']
X = df[features]
y = df['ArrDel15']

# Drop rows with missing values in features or target
df_clean = pd.concat([X, y], axis=1).dropna()
X = df_clean[features]
y = df_clean['ArrDel15']

# One-hot encode categorical features (Reporting_Airline, Origin, Dest, DayOfWeek)
# Keep all dummy variables (drop_first=False)
X_encoded = pd.get_dummies(X, columns=['Reporting_Airline', 'Origin', 'Dest', 'DayOfWeek'], drop_first=False)

# Stratified 75/25 split with random_state=0
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, stratify=y, random_state=0
)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Baseline accuracy: always predict "not delayed" (class 0)
baseline_accuracy = (y_test == 0).mean()

print(f"Test set accuracy (logistic regression): {test_accuracy:.4f}")
print(f"Baseline accuracy (always predict not delayed): {baseline_accuracy:.4f}")