import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "C:\Users\aruna\OneDrive\Desktop\Major-Pro\DATASET\Raw\final_data.csv"   # Change path if needed
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop(columns=["player", "team", "name", "position"])

# Define target and features
X = df.drop(columns=["current_value", "highest_value"])  # highest_value removed to avoid leakage
y = df["current_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Dataset processed successfully")
print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)
