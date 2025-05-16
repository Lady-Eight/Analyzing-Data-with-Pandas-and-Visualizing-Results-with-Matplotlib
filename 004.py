# Data Analysis and Visualization with Pandas and Matplotlib


### Code
import pandas as pd

# Load the dataset with error handling
try:
    df = pd.read_csv('iris.csv')
    print("Dataset loaded successfully.")
    print("First few rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'iris.csv' not found. Please download it from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data and save as 'iris.csv' in the current directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Check data types
print("\nData types of each column:")
print(df.dtypes)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())
## Task 2: Basic Data Analysis

### Code
# Summary statistics for numerical columns
print("Summary statistics of numerical columns:")
print(df.describe())

# Group by species and compute the mean
grouped = df.groupby('species').mean()
print("\nMean measurements by species:")
print(grouped)

## Task 3: Data Visualization

### Code
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")

#### 1. Line Chart: Measurements for Setosa Species
setosa = df[df['species'] == 'setosa']
plt.figure(figsize=(10, 6))
plt.plot(setosa.index, setosa['sepal_length'], label='Sepal Length', linestyle='-', marker='o')
plt.plot(setosa.index, setosa['petal_length'], label='Petal Length', linestyle='--', marker='x')
plt.title('Sepal and Petal Lengths for Setosa Species')
plt.xlabel('Sample Index')
plt.ylabel('Length (cm)')
plt.legend()
plt.tight_layout()  # Ensure labels fit in saved image
plt.savefig('setosa_line_chart.png')

#### 2. Bar Chart: Average Measurements by Species
plt.figure(figsize=(10, 6))
grouped.plot(kind='bar')
plt.title('Average Measurements by Species')
plt.xlabel('Species')
plt.ylabel('Measurement (cm)')
plt.legend(title='Measurement Type')
plt.tight_layout()  # Ensure labels fit in saved image
plt.savefig('species_bar_chart.png')

#### 3. Histogram: Distribution of Sepal Width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal_width'], bins=15, color='lightgreen', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()  # Ensure labels fit in saved image
plt.savefig('sepal_width_histogram.png')

#### 4. Scatter Plot: Petal Length vs. Petal Width
plt.figure(figsize=(8, 5))
sns.scatterplot(x='petal_length', y='petal_width', hue='species', size='sepal_length', data=df)
plt.title('Petal Length vs. Petal Width by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species')
plt.tight_layout()  # Ensure labels fit in saved image
plt.savefig('petal_scatter_plot.png')

