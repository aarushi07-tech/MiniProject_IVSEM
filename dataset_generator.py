import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

study_hours = np.random.normal(4,1.5,n).clip(0,10)
sleep_hours = np.random.normal(7,1,n).clip(4,10)
attendance = np.random.normal(80,10,n).clip(40,100)
internet_usage = np.random.normal(3,1,n).clip(0,8)

stress_level = 10 - sleep_hours + np.random.normal(0,1,n)

final_score = (
    study_hours * 10 +
    attendance * 0.3 -
    internet_usage * 2 -
    stress_level * 2 +
    np.random.normal(0,5,n)
)

final_score = np.clip(final_score,0,100)

result = (final_score >= 40).astype(int)

data = pd.DataFrame({
    "study_hours":study_hours,
    "sleep_hours":sleep_hours,
    "attendance":attendance,
    "internet_usage":internet_usage,
    "stress_level":stress_level,
    "final_score":final_score,
    "result":result
})

data.to_csv("data/student_data.csv",index=False)

print("Dataset generated successfully")
