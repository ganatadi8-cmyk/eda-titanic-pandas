import pandas as pd
import seaborn as sns

# 1. Load the dataset
df = sns.load_dataset('titanic')

# ---------------------------------------------------------
# KEY FEATURE 1: DataFrame operations (head, info, describe)
# ---------------------------------------------------------
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
df.info()

print("\n--- Summary Statistics ---")
print(df.describe(include='all'))

# ---------------------------------------------------------
# KEY FEATURE 2: Handling missing values and duplicates
# ---------------------------------------------------------
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# Check and remove duplicates
print(f"\nDuplicates found: {df.duplicated().sum()}")
df = df.drop_duplicates()

# Handle missing values
# Fill numeric missing values (Age) with the median
df['age'] = df['age'].fillna(df['age'].median())

# Fill categorical missing values (Embarked) with the mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop 'deck' column if it has too many missing values
if 'deck' in df.columns:
    df = df.drop(columns=['deck'])

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# ---------------------------------------------------------
# KEY FEATURE 3: Grouping and aggregation analysis
# ---------------------------------------------------------
# Survival rate by passenger class (pclass)
print("\n--- Survival Rate by Class ---")
class_survival = df.groupby('pclass')['survived'].agg(['mean', 'count'])
print(class_survival)

# Survival rate by gender (sex)
print("\n--- Survival Rate by Gender ---")
sex_survival = df.groupby('sex')['survived'].agg(['mean', 'count'])
print(sex_survival)

# Average fare paid by embarkation town and passenger class
print("\n--- Average Fare by Town & Class ---")
fare_analysis = df.groupby(['embark_town', 'pclass'])['fare'].mean()
print(fare_analysis)