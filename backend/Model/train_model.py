import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("../data/student_data.csv")
X = data[[
    "study_hours",
    "sleep_hours",
    "attendance",
    "internet_usage",
    "stress_level"
]]

y = data["result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

joblib.dump(model,"model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("Model trained and saved successfully")
