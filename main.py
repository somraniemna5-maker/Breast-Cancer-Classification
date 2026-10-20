"""Mini projet Data Science : classification du cancer du sein (tumeur bénigne / maligne).

Pipeline : chargement -> EDA -> comparaison de modèles (validation croisée)
-> évaluation sur test -> importance des variables.
Usage : python main.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # Dans le dataset : 0 = malin, 1 = bénin
    df["diagnosis"] = df["target"].map({0: "malin", 1: "bénin"})
    return df, data.feature_names


def eda(df, features):
    print("Dimensions :", df.shape)
    print("Valeurs manquantes :", int(df.isna().sum().sum()))
    print("\nRépartition des classes :\n", df["diagnosis"].value_counts())

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="diagnosis", hue="diagnosis", legend=False)
    plt.title("Répartition des classes")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_classes.png", dpi=120)
    plt.close()

    top = df[list(features)].corrwith(df["target"]).abs().nlargest(8).index
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[list(top)].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Corrélations (8 variables les plus liées à la cible)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_correlations.png", dpi=120)
    plt.close()


def compare_models(X_train, y_train):
    models = {
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = scores
        print(f"{name:<24} AUC CV = {scores.mean():.3f} ± {scores.std():.3f}")

    plt.figure(figsize=(8, 4))
    plt.boxplot(results.values(), tick_labels=results.keys())
    plt.ylabel("AUC (validation croisée)")
    plt.title("Comparaison des modèles")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_comparaison_modeles.png", dpi=120)
    plt.close()

    best = max(results, key=lambda k: results[k].mean())
    return best, models[best]


def evaluate(model, name, X_train, X_test, y_train, y_test, features):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== Meilleur modèle : {name} ===")
    print(classification_report(y_test, pred, target_names=["malin", "bénin"]))
    print(f"AUC test : {roc_auc_score(y_test, proba):.3f}")

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["malin", "bénin"], cmap="Blues")
    plt.title(f"Matrice de confusion – {name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_matrice_confusion.png", dpi=120)
    plt.close()

    imp = permutation_importance(model, X_test, y_test, n_repeats=20,
                                 random_state=SEED, scoring="roc_auc")
    s = pd.Series(imp.importances_mean, index=features).nlargest(10)[::-1]
    plt.figure(figsize=(8, 5))
    s.plot.barh()
    plt.xlabel("Baisse d'AUC (importance par permutation)")
    plt.title("Top 10 des variables importantes")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_importance_variables.png", dpi=120)
    plt.close()
    print("\nTop 5 variables :", ", ".join(s.index[::-1][:5]))


def main():
    df, features = load_data()
    eda(df, features)
    X, y = df[list(features)], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print("\n--- Validation croisée (5 plis) ---")
    name, model = compare_models(X_train, y_train)
    evaluate(model, name, X_train, X_test, y_train, y_test, features)
    print(f"\nFigures enregistrées dans {FIG_DIR}/")


if __name__ == "__main__":
    main()
