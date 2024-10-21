import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
iris_data = pd.read_csv(url, header=None, names=column_names)

# Display the first few rows of the dataset
print(iris_data.head())

# Basic statistics
print(iris_data.describe())

# Pairplot
sns.pairplot(iris_data, hue='Species')
plt.title('Iris Pairplot')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(iris_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Species', y='PetalLength', data=iris_data)
plt.title('Boxplot of Petal Length by Species')
plt.show()
