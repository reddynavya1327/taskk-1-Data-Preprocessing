
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Titanic dataset directly from URL
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column because it has many missing values
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# Drop any remaining missing rows
df.dropna(inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Encode categorical columns
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

print("\nData after encoding:")
print(df.head())

# Standardize numerical columns
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

print("\nData after scaling:")
print(df.head())

# Boxplots for outlier detection
plt.figure(figsize=(6, 4))
plt.boxplot(df["Age"])
plt.title("Boxplot of Age")
plt.ylabel("Age")
plt.show()

plt.figure(figsize=(6, 4))
plt.boxplot(df["Fare"])
plt.title("Boxplot of Fare")
plt.ylabel("Fare")
plt.show()

# Remove outliers using IQR method
for col in ["Age", "Fare"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

print("\nShape after removing outliers:", df.shape)

# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nPreprocessing completed successfully.")
print("Cleaned file saved as: cleaned_titanic.csv")
