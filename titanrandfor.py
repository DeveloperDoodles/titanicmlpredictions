import pandas as pd
from sklearn.ensemble import RandomForestClassifier

traindata = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")

titanicfeat = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

traindata = traindata.dropna(subset=titanicfeat)
testdata = testdata.dropna(subset=titanicfeat)

traindata['Sex'] = traindata['Sex'].map({'male': 0, 'female': 1})
testdata['Sex'] = testdata['Sex'].map({'male': 0, 'female': 1})
y = traindata.Survived
X = traindata[titanicfeat]
test_X = testdata[titanicfeat]

titanicmodel = RandomForestClassifier(random_state=1)

titanicmodel.fit(X, y)

predictions = titanicmodel.predict(test_X)

print("The predictions are:")
print(predictions)