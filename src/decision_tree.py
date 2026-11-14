import pandas as pd
from sklearn.tree import DecisionTreeClassifier

traindata = pd.read_csv("../data/train.csv")
testdata = pd.read_csv("../data/test.csv")

titanicfeat = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

traindata = traindata.dropna(subset=titanicfeat)
testdata = testdata.dropna(subset=titanicfeat)

traindata['Sex'] = traindata['Sex'].map({'male': 0, 'female': 1})
testdata['Sex'] = testdata['Sex'].map({'male': 0, 'female': 1})

y = traindata.Survived

X = traindata[titanicfeat]
test_X = testdata[titanicfeat]

titanicmodel = DecisionTreeClassifier(random_state=1)
titanicmodel.fit(X, y)
predictions = titanicmodel.predict(test_X)
print("The predictions are:")
print(predictions)