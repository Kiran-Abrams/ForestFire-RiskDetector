import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Select 5 features + target
X = df[["lat", "lon", "temp_mean", "humidity_min", "wind_speed_max"]]
y = df["occured"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save new model
joblib.dump(model, "final_model.pkl")

print("✅ Model retrained with 5 features and saved as final_model.pkl")
