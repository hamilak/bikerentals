import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv(filepath_or_buffer='bike-sharing-dataset/hour.csv')

# print(df.head(10))
# print(df.shape)
# print(df.info())
# print(df.isnull().sum())
# sns.heatmap(df.isnull(), annot=True, cmap="viridis")
# plt.show()

# sns.set(style='whitegrid')
# sns.scatterplot(data=df)
# plt.show()

# corrMatrix = df.corr()
# sns.heatmap(corrMatrix , annot=True, cmap="viridis")
# plt.show()

# sns.pairplot(df)
# plt.show()

to_drop = ['instant', 'dteday', 'casual', 'registered']
df.drop(to_drop, inplace=True, axis=1)

# X = df.drop(['cnt'], axis=1)
# Y = df['cnt']

X = df.drop(['cnt', 'holiday', 'workingday', 'weathersit'], axis=1)
Y = df['cnt']

# st_x = StandardScaler()
# X = st_x.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)
training_data_prediction = lin_reg_model.predict(X_train)

train_error_score = metrics.r2_score(Y_train, training_data_prediction)

print("R squared error - Training:", train_error_score)

Y_pred = lin_reg_model.predict(X_test)

test_error_score = metrics.r2_score(Y_test, Y_pred)
print("R squared error - Test:", test_error_score)

# plt.scatter(Y_test, Y_pred)
# plt.show()
# plt.hist(Y_test - Y_pred)
# plt.show()
# sns.regplot(x=Y_test, y=Y_pred, scatter_kws={"color":"green"}, line_kws={"color":"red"})
# plt.show()

# plt.hist(df)
# plt.show()

sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(df['cnt'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1})
plt.show()


# Random forest regression

rf_classifier = RandomForestClassifier(n_estimators=50, random_state=0, criterion='entropy')
rf_classifier.fit(X_train, Y_train)
y_pred = rf_classifier.predict(X_test)
#
print("Accuracy:", accuracy_score(Y_test, y_pred))
#
# feature_importances_df = pd.DataFrame(
#     {"feature": list(X.columns), "importance": rf_classifier.feature_importances_}
# ).sort_values("importance", ascending=False)
#
# print(feature_importances_df)

# sns.barplot(x=feature_importances_df.feature,y=feature_importances_df.importance)
#
# plt.xlabel("Feature importance score")
# plt.ylabel("Features")
# plt.title("Visualizing Important Features")
# plt.xticks(
#     rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large"
# )
# plt.show()

# Neural networks

