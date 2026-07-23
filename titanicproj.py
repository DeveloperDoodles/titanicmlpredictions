import pandas as pd
from sklearn.linear_model import LogisticRegression

traindata = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")

traindata['Age'] = traindata['Age'].fillna(traindata['Age'].median())
testdata['Age'] = testdata['Age'].fillna(traindata['Age'].median())

traindata['Fare'] = traindata['Fare'].fillna(traindata['Fare'].median())
testdata['Fare'] = testdata['Fare'].fillna(traindata['Fare'].median())

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
count0 = count1 = 0
for i in predictions:
    if i == 0:
        count0 += 1
    else :
        count1 += 1

print(count0, count1)
