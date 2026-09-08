"""
Day 5, Week 2: scikit-learn pipeline exercise.

Dataset: sklearn's built-in breast cancer dataset — a binary
classification problem (malignant/benign) with 30 numeric features per
sample. Chosen because it's tabular, not image data — a deliberately
different problem shape from both your ISL project and the PyTorch
exercise above, since sklearn's real strength is classical ML on
structured/tabular data, not deep learning.

The point of this file isn't the dataset — it's the *pipeline pattern*:
sklearn.pipeline.Pipeline chains preprocessing + model into one object,
so you never accidentally fit a scaler on test data (a real, common bug
called "data leakage" — worth being able to explain if asked).

Run: python day5_sklearn_pipeline.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# 1. Load and split data
# ---------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target  # y: 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# 2. Build pipelines — preprocessing + model bundled as ONE object.
#
#    Why this matters (data leakage, concretely): if you called
#    scaler.fit(X_train_and_test_combined) before splitting, the scaler
#    would "see" statistics from the test set during training — an
#    optimistic bias that inflates your reported accuracy and won't
#    hold up on genuinely new data. Pipeline.fit() only ever calls
#    .fit() on the training fold; .predict() reuses those already-fit
#    parameters on new data. This is the mechanism, not just the rule.
# ---------------------------------------------------------------------
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # LogisticRegression needs scaled features to converge well
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

forest_pipeline = Pipeline([
    # No scaler needed — tree-based models split on raw feature
    # thresholds, so feature scale doesn't affect them. Including a
    # scaler here wouldn't break anything, just wouldn't help either;
    # worth knowing WHY, not just copying the pattern from the other pipeline.
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

models = {"Logistic Regression": logistic_pipeline, "Random Forest": forest_pipeline}

# ---------------------------------------------------------------------
# 3. Cross-validation — a single train/test split can mislead you
#    (Day 2 revision topic, now hands-on). 5-fold CV trains/evaluates
#    5 times on different splits and reports the spread, not just one
#    number.
# ---------------------------------------------------------------------
for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"{name} — 5-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# ---------------------------------------------------------------------
# 4. Fit on full training set, evaluate on the held-out test set
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n{name} — test set results:")
    print(classification_report(y_test, predictions, target_names=data.target_names))

    auc = roc_auc_score(y_test, probabilities)
    print(f"ROC-AUC: {auc:.3f}")

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

# ---------------------------------------------------------------------
# Self-check: for a medical screening dataset like this, which matters
# more — precision or recall, for the "malignant" class specifically?
# (Hint: a false negative here means telling someone with cancer they're
# fine. A false positive means an unnecessary follow-up test. Those are
# NOT equally costly — that asymmetry is why "just look at accuracy"
# is often the wrong instinct in real applications, tying directly back
# to your Day 2 revision on evaluation metrics.)
# ---------------------------------------------------------------------
