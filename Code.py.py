# 1. Load and Explore the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv("NYC_Wi-Fi_Hotspot_Locations.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Show basic information about the dataset
print("Dataset Info")
print(df.info())
print("\nFirst 5 Rows")
print(df.head())
print("\nColumn Names")
print(df.columns.tolist())

# 2. Clean and Preprocess the Data
# Drop columns that are not useful if they exist
columns_to_drop = ['X', 'Provider']
for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Remove rows that have missing latitude or longitude
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Show missing values after cleaning
print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# 3. Filter Data Based on Conditions

# A. Find hotspots located in parks
parks = df[df['Location'].str.contains("Park", case=False, na=False)]
print("\nHotspots in Parks")
print(parks[['Name', 'Location', 'Borough Name']].head())

# B. Find hotspots that do not have a name
missing_name = df[df['Name'].isnull()]
print("\nHotspots Without a Name")
print(missing_name[['Location', 'Type']].head())

# Bar chart: Hotspots with and without names
has_name = df['Name'].notnull().sum()
no_name = df['Name'].isnull().sum()

plt.figure(figsize=(6, 4))
plt.bar(['Has Name', 'Missing Name'], [has_name, no_name], color=['skyblue', 'orange'])
plt.title("Hotspots With and Without Names")
plt.ylabel("Number of Hotspots")
plt.tight_layout()
plt.show()

# C. Find hotspots in the Bronx by SPECTRUM offering Limited Free service
if 'Provider' in df.columns:
    bronx_spectrum = df[
        (df["Type"].str.contains("Limited Free", case=False, na=False)) &
        (df["Borough Name"].str.upper() == "BRONX") &
        (df["Provider"] == "SPECTRUM")
    ]
    print("\nSPECTRUM Hotspots in Bronx")
    print(bronx_spectrum[['Name', 'Location', 'Type']])
else:
    bronx_spectrum = pd.DataFrame()
    print("\nColumn 'Provider' not found in the dataset.")

# 4. Analyze Data with Statistics

# Show summary statistics for Latitude and Longitude
print("\nSummary of Latitude and Longitude")
print(df[["Latitude", "Longitude"]].describe())

# Show average latitude and longitude per borough
print("\nAverage Latitude and Longitude per Borough")
print(df.groupby("Borough Name")[["Latitude", "Longitude"]].mean())

# Histogram: Distribution of Latitude
plt.hist(df['Latitude'], bins=20, color='lightgreen', edgecolor='black')
plt.title("Distribution of Latitude")
plt.xlabel("Latitude")
plt.ylabel("Number of Hotspots")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot: Distribution of Longitude
plt.boxplot(df['Longitude'], vert=False)
plt.title("Boxplot of Longitude")
plt.xlabel("Longitude")
plt.tight_layout()
plt.show()

# 5. Visualize the Data

# Bar chart: Number of Hotspots in each Borough
plt.figure(figsize=(8, 5))
sns.countplot(x='Borough Name', data=df, palette='pastel')
plt.title("Hotspots in Each Borough")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie chart: Types of Hotspot Services
plt.figure(figsize=(6, 6))
df['Type'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title("Types of Wi-Fi Hotspots")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Scatter plot: Location of All Hotspots
plt.figure(figsize=(10, 6))
plt.scatter(df['Longitude'], df['Latitude'], alpha=0.6, color='green')
plt.title("Location of Wi-Fi Hotspots in NYC")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Heatmap: Correlation of Numeric Columns heat map
plt.figure(figsize=(10, 6))
numeric_columns = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_columns.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numeric Columns heat map")
plt.tight_layout()
plt.show()

# 6. Save Filtered Output

# Save Bronx SPECTRUM data if available
if not bronx_spectrum.empty:
    bronx_spectrum.to_csv("bronx_spectrum_wifi.csv", index=False)
    print("\nFiltered Bronx SPECTRUM hotspots saved to 'bronx_spectrum_wifi.csv'")
else:
    print("\nNo matching Bronx SPECTRUM data to save.")
