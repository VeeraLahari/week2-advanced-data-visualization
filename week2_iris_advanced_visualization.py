# Week 2: Advanced Data Visualization and Storytelling with Python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates

sns.set_theme(style="whitegrid")

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns=["target"])
df = df.rename(columns={
    "sepal length (cm)": "Sepal Length",
    "sepal width (cm)": "Sepal Width",
    "petal length (cm)": "Petal Length",
    "petal width (cm)": "Petal Width"
})

# 1. Pair plot
sns.pairplot(df, hue="species", corner=True, diag_kind="hist")
plt.suptitle("Pair Plot of Iris Measurements by Species", y=1.02)
plt.show()

# 2. Violin plot
sns.violinplot(data=df, x="species", y="Petal Length", inner="box")
plt.title("Petal Length Distribution by Species")
plt.show()

# 3. Correlation heatmap
corr = df.drop(columns=["species"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", square=True)
plt.title("Correlation Heatmap of Iris Measurements")
plt.show()

# 4. Box plot
long_df = df.melt(id_vars="species", var_name="Measurement", value_name="Value")
sns.boxplot(data=long_df, x="Measurement", y="Value", hue="species")
plt.title("Measurement Variation Across Iris Species")
plt.show()

# 5. Scatter plot with trend
sns.scatterplot(data=df, x="Petal Length", y="Petal Width", hue="species", s=65)
sns.regplot(data=df, x="Petal Length", y="Petal Width", scatter=False, color="black")
plt.title("Petal Length vs Petal Width with Overall Trend")
plt.show()

# 6. Parallel coordinates
features = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
scaled = df.copy()
scaled[features] = StandardScaler().fit_transform(scaled[features])
parallel_coordinates(scaled[["species"] + features], "species", alpha=0.35)
plt.title("Parallel Coordinates: Standardized Iris Profiles")
plt.show()
