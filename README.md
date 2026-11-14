# 🚢 Titanic Survival Predictions

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)

Predicting which passengers survived the Titanic disaster, using four classic classifiers. This is a beginner-friendly comparison of how different preprocessing choices and models affect predictions on the same data.

---

## 📋 The problem

**Given** a passenger's class, sex, age, family aboard, and fare,
**predict** whether they survived (`1`) or not (`0`).

| | |
|---|---|
| **Problem type** | Binary classification |
| **Training data** | `data/train.csv`: 891 passengers, with the `Survived` label |
| **Test data** | `data/test.csv`: 418 passengers, no labels |
| **Features used** | `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare` |

## 🧠 Approaches

Each script trains one model on the same six features and prints predictions for the test passengers.

| Script | Model | Missing values | Notes |
|---|---|---|---|
| `src/logistic_regression_imputed.py` | Logistic Regression | Fill `Age` and `Fare` with the median | Keeps every passenger |
| `src/logistic_regression_dropna.py` | Logistic Regression | Drop rows with any missing value | Fewer training *and* test passengers |
| `src/decision_tree.py` | Decision Tree | Drop rows missing a feature | Fewer test passengers predicted |
| `src/random_forest.py` | Random Forest | Drop rows missing a feature | Fewer test passengers predicted |

**Why it matters:** the same model can give different predictions depending on how missing data is handled, and dropping rows throws away information. Comparing the two logistic regression scripts shows that effect directly.

## 🚀 Run it

Requires Python 3.10+ with `pandas` and `scikit-learn`.

```bash
pip install pandas scikit-learn
cd src
python random_forest.py
```

Each script prints the predicted survival for each test passenger, as `0` or `1`. `logistic_regression_imputed.py` also prints how many passengers are predicted to survive and how many are not.

> The scripts read the CSVs from `../data/`, so run them from inside `src/`.

## 📁 Repository layout

```
titanicmlpredictions/
├── data/
│   ├── train.csv               Labeled passengers (for training)
│   └── test.csv                Unlabeled passengers (for prediction)
├── src/
│   ├── logistic_regression_imputed.py
│   ├── logistic_regression_dropna.py
│   ├── decision_tree.py
│   └── random_forest.py
└── README.md
```

## 🔮 Next steps

- [ ] Hold out part of `train.csv` to measure accuracy. `test.csv` has no labels, so the scripts can't report their own accuracy yet.
- [ ] Add cross-validation to compare the four models fairly
- [ ] Engineer features such as family size, or extract titles like "Mr." and "Mrs." from `Name`
- [ ] Submit predictions to the Kaggle Titanic competition for an official score

---

<p align="center"><sub>Built during the Summer 2026 internship · Python · scikit-learn · pandas</sub></p>
