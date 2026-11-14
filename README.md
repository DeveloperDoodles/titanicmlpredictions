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
│   ├── random_forest.py
│   └── evaluate_models.py      Cross-validated comparison, base vs engineered features
└── README.md
```

## 📊 Results

`test.csv` has no labels, so these scores come from **5-fold stratified cross-validation on `train.csv`**. Run `src/evaluate_models.py` to reproduce them.

| Model | Base features | + Engineered features |
|---|:---:|:---:|
| Logistic Regression | 0.788 ± 0.011 | 0.807 ± 0.018 |
| Decision Tree | 0.796 ± 0.036 | 0.799 ± 0.008 |
| **Random Forest** | **0.825 ± 0.023** | **0.826 ± 0.011** |

*Mean accuracy across 5 folds, ± standard deviation.*

**Takeaways**
- **Random Forest is the strongest model** on this data, and it's also the most stable on the engineered features.
- **Feature engineering helps Logistic Regression** (about +2 points). It makes little difference for the tree models, which can already capture those interactions.

**Engineered features** (`src/evaluate_models.py`):
- `FamilySize`: siblings + spouses + parents + children + 1
- `IsAlone`: 1 if the passenger travelled alone
- `Title`: the honorific from `Name` (Mr, Mrs, Miss, Master, or Rare)

## 🔮 Next steps

- [x] Measure accuracy on the training data (cross-validation)
- [x] Compare the four models fairly with the same folds
- [x] Engineer features from family size and titles
- [ ] Submit predictions to the Kaggle Titanic competition for an official score
- [ ] Try hyperparameter tuning for the Random Forest

---

<p align="center"><sub>Built during the Summer 2026 internship · Python · scikit-learn · pandas</sub></p>
