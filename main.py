import pandas as pd
df = pd.read_csv('heart.csv')
filtered_df = df[(df["Sex"] == "F") & (df["Age"].between(50, 60))]
mean_cholesterol = filtered_df["Cholesterol"].mean()
