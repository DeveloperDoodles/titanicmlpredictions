"""Compare the four classifiers with 5-fold cross-validation on train.csv.

Runs two feature sets so you can see whether feature engineering helps:
  - base:        the six original features
  - engineered:  base + FamilySize, IsAlone, and Title

The test set has no labels, so every score here comes from the training data.
Run from inside src/:  python evaluate_models.py
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

TRAIN_PATH = "../data/train.csv"

BASE_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
ENGINEERED_FEATURES = BASE_FEATURES + ["FamilySize", "IsAlone", "Title"]

# Titles that appear often enough to keep; everything else becomes 4 ("Rare").
TITLE_CODES = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3}


def load_training_data():
    df = pd.read_csv(TRAIN_PATH)

    # Same cleaning as the imputed logistic regression script.
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Feature engineering.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    titles = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = titles.map(TITLE_CODES).fillna(4).astype(int)

    return df


def make_models():
    # Same model settings as the scripts in this folder.
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=1),
        "Random Forest": RandomForestClassifier(random_state=1),
    }


def main():
    df = load_training_data()
    y = df["Survived"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    feature_sets = {"base": BASE_FEATURES, "engineered": ENGINEERED_FEATURES}

    print(f"{'Model':<22}{'Features':<12}{'Mean acc':>10}{'Std':>8}")
    print("-" * 52)
    for name, model in make_models().items():
        for set_name, features in feature_sets.items():
            scores = cross_val_score(model, df[features], y, cv=cv, scoring="accuracy")
            print(f"{name:<22}{set_name:<12}{scores.mean():>10.4f}{scores.std():>8.4f}")


if __name__ == "__main__":
    main()
