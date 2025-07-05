import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the raw dataset (adjust path if needed)
df = pd.read_csv("winequality-red-selected-missing.csv")

# Display basic info
df.info()

# Handle missing values (drop rows with missing data)
df.dropna(inplace=True)

# Create binary target column: 1 = Good (>=7), 0 = Not Good
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
df.drop(columns=['quality'], inplace=True)

# Feature and target separation
X = df.drop(columns=['quality_label'])
y = df['quality_label']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and scaler
pickle.dump(model, open("wine_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))