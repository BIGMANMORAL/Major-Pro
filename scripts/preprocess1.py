import pandas as pd

# Load dataset
path = "DATASET\Raw\RAW_dataset.csv.csv"
df = pd.read_csv(path)

# Drop irrelevant columns (like URLs, names, redundant encodings)
irrelevant_cols = [
    "player",      # URL info
    "name",        # player name (not numerical feature)
    "highest_value"  # could cause data leakage (past market value)
]

df_cleaned = df.drop(columns=irrelevant_cols, errors="ignore")

# Handle missing values (drop or fill)
df_cleaned = df_cleaned.dropna()  # Option: drop rows with missing data
# Alternatively: df_cleaned = df_cleaned.fillna(0)

# Check data types
print(df_cleaned.info())

# Save cleaned dataset
df_cleaned.to_csv("DATASET\Processed\cleaned_final_data.csv", index=False)

print(" Cleaned dataset saved as 'cleaned_final_data.csv'")
