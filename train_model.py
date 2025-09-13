import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("final_dataset.csv")  # replace with your actual dataset name
print("✅ Columns in dataset:", df.columns.tolist())

# Select only the most important features (to keep model light)
features = ["daynight_N", "lat", "lon", "temp_mean", "humidity_min", "wind_speed_max"]
X = df[features]
y = df["occured"]  # target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train smaller RandomForest
model = RandomForestClassifier(
    n_estimators=30,   # fewer trees → smaller size
    max_depth=8,       # limit depth → smaller trees
    random_state=42
)
model.fit(X_train, y_train)

# Save compressed model
joblib.dump(model, "final_model.pkl", compress=3)

print("🔥 Model retrained and saved as compressed final_model.pkl")
