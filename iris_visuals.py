import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================
# Task 1: Load and Explore Data
# =============================
# Load the Iris dataset directly from seaborn
df = sns.load_dataset("iris")

# Display first few rows
print("---- First 5 rows ----")
print(df.head())

# Check data types and missing values
print("\n---- Info ----")
print(df.info())

print("\n---- Missing Values ----")
print(df.isnull().sum())

# Clean dataset (if any missing)
df = df.dropna()

# =============================
# Task 2: Basic Data Analysis
# =============================
print("\n---- Statistical Summary ----")
print(df.describe())

# Group by species and get mean of numeric columns
grouped = df.groupby("species").mean(numeric_only=True)
print("\n---- Mean values per species ----")
print(grouped)

# =============================
# Task 3: Data Visualizations
# =============================
# Create 'plots' folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# --- 1. Line chart (cumulative mean of sepal_length over rows) ---
plt.figure(figsize=(8,5))
df['sepal_length'].expanding().mean().plot(title="Cumulative Mean of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.savefig("plots/line_chart.png")
plt.show()

# --- 2. Bar chart (average petal length per species) ---
plt.figure(figsize=(8,5))
df.groupby("species")["petal_length"].mean().plot(kind='bar', color='skyblue')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.savefig("plots/bar_chart.png")
plt.show()

# --- 3. Histogram (distribution of sepal_width) ---
plt.figure(figsize=(8,5))
plt.hist(df["sepal_width"], bins=20, color='lightgreen', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.savefig("plots/histogram.png")
plt.show()

# --- 4. Scatter plot (sepal_length vs petal_length) ---
plt.figure(figsize=(8,5))
plt.scatter(df["sepal_length"], df["petal_length"], c='purple')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.savefig("plots/scatter_plot.png")
plt.show()

print("\n✅ All tasks completed! Charts saved inside the 'plots' folder.")
