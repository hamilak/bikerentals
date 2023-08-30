import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(filepath_or_buffer='bike-sharing-dataset/hour.csv')

print(df.head(10))
print(df.shape)
print(df.info())
print(df.isnull().sum())
sns.heatmap(df.isnull(), annot=True, cmap="viridis")
plt.show()

sns.set(style='whitegrid')
sns.scatterplot(data=df)
plt.show()

corrMatrix = df.corr()
sns.heatmap(corrMatrix , annot=True, cmap="viridis")
plt.show()

sns.pairplot(df)
plt.show()
