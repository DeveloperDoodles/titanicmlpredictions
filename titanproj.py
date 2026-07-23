import pandas as pd
from sklearn.linear_model import LogisticRegression

traindata = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")

traindata = traindata.dropna()
testdata = testdata.dropna()

traindata['Sex'] = traindata['Sex'].map({'male': 0, 'female': 1})
testdata['Sex'] = testdata['Sex'].map({'male': 0, 'female': 1})

y = traindata.Survived

titanicfeat = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

X = traindata[titanicfeat]
test_X = testdata[titanicfeat]

titanicmodel = LogisticRegression(max_iter=1000)

titanicmodel.fit(X, y)

predictions = titanicmodel.predict(test_X)

print("The predictions are:")
print(predictions)