import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from evaluation import evaluate_metrics

df = pd.read_csv('bike-sharing-dataset/hour.csv')

to_drop = ['instant', 'dteday', 'casual', 'registered']
df.drop(to_drop, inplace=True, axis=1)

# X = df.drop(['cnt'], axis=1)
# Y = df['cnt']

X = df.drop(['cnt', 'holiday', 'workingday', 'weathersit'], axis=1)
Y = df['cnt']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train_scaled, Y_train)

training_data_prediction = lin_reg_model.predict(X_train_scaled)
evaluate_metrics(Y_train, training_data_prediction)

# test data accuracy
Y_pred = lin_reg_model.predict(X_test_scaled)
evaluate_metrics(Y_test, Y_pred)

# plt.scatter(Y_test, Y_pred)
# plt.show()
# plt.hist(Y_test - Y_pred)
# plt.show()
# sns.regplot(x=Y_test, y=Y_pred, scatter_kws={"color":"green"}, line_kws={"color":"red"})
# plt.show()

# plt.hist(df)
# plt.show()

# sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
# sns.distplot(df['cnt'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1})
# plt.show()

# Random forest regression
rf_classifier = RandomForestRegressor(n_estimators=50, random_state=0, criterion='squared_error')
rf_classifier.fit(X_train_scaled, Y_train)
y_pred = rf_classifier.predict(X_test_scaled)

evaluate_metrics(Y_test, y_pred)

# Check feature importance to improve the model
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


