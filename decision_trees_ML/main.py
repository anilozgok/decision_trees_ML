import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# load and check data
dataset = pd.read_csv("DecisionTrees_titanic.csv")
# print(dataset.head())

# train and test sets
X = dataset.drop("Survived", axis=1)
y = dataset.loc[:, "Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# training model using decision tree
model_1 = DecisionTreeClassifier()
model_1.fit(X_train, y_train)

# training model using random forest
model_2 = RandomForestClassifier()
model_2.fit(X_train, y_train)

# making predictions
decisionTree_Predictions = model_1.predict(X_test)
randomForest_Predictions = model_2.predict(X_test)

# classification report comparison
print(classification_report(y_test, decisionTree_Predictions))
print()
print(classification_report(y_test, randomForest_Predictions))

# visualizing decision tree
fig = plt.figure(figsize=(10, 10))
plot_tree(model_1,
          max_depth=4,
          feature_names=X.columns,
          filled=True,
          impurity=False,
          rounded=True,
          precision=1)
plt.show()
