# LAB 3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.metrics.pairwise import euclidean_distances

# ---- User config ----
DATA_PATH = r"C:/Users/aksha/Downloads/reduced_dataset-release.csv"  # <-- set path to your CSV here
RANDOM_STATE = 42

# ---- Functions (all in single block) ----
def load_and_clean(path, feature_cols, target_col='1_DAY_RETURN'):
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path, low_memory=False)
    # Keep only required columns if present
    required = [c for c in feature_cols + [target_col] if c in df.columns]
    df = df[required].copy()
    # Coerce feature columns to numeric, invalid -> NaN
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    # Drop rows with NaN in any required column
    df.dropna(axis=0, subset=feature_cols + [target_col], inplace=True)
    # Create binary class: 1 if return > 0 else 0
    df['Class'] = (df[target_col] > 0).astype(int)
    return df

def compute_intrainter_inter(X, y, sample_n=50):
    X0 = X[y==0]
    X1 = X[y==1]
    n0 = min(sample_n, len(X0))
    n1 = min(sample_n, len(X1))
    if n0 < 2 or n1 < 2:
        raise ValueError("Not enough samples in one of the classes for distance computations.")
    X0s = X0.sample(n=n0, random_state=RANDOM_STATE)
    X1s = X1.sample(n=n1, random_state=RANDOM_STATE)
    d0 = euclidean_distances(X0s)
    intr0 = d0[np.triu_indices_from(d0, k=1)]
    d1 = euclidean_distances(X1s)
    intr1 = d1[np.triu_indices_from(d1, k=1)]
    inter = euclidean_distances(X0s, X1s).flatten()
    return intr0, intr1, inter

def plot_distance_distributions(intr0, intr1, inter):
    plt.figure(figsize=(8,5))
    plt.hist(intr0, bins=30, alpha=0.5, label='Intra-class (0)')
    plt.hist(intr1, bins=30, alpha=0.5, label='Intra-class (1)')
    plt.hist(inter, bins=30, alpha=0.5, label='Inter-class')
    plt.title("Distance Distributions")
    plt.xlabel("Euclidean distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_hist_feature(series, bins=30, title=None):
    plt.figure(figsize=(7,4))
    sns.histplot(series, bins=bins, stat='count')
    plt.title(title or f"Histogram of {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.show()
    return series.mean(), series.var()

def minkowski_distance_between_two(vecA, vecB, r_values=range(1,11)):
    dists = []
    for r in r_values:
        d = np.sum(np.abs(vecA - vecB)**r)**(1.0/r)
        dists.append(d)
    return list(r_values), dists

def plot_minkowski(r_vals, distances):
    plt.figure(figsize=(7,4))
    plt.plot(r_vals, distances, marker='o')
    plt.title("Minkowski Distance (r values)")
    plt.xlabel("r")
    plt.ylabel("Distance")
    plt.xticks(r_vals)
    plt.grid(True)
    plt.show()

def split_data(X, y, test_size=0.3):
    return sk_train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

def train_knn(X_train, y_train, k=3, **kwargs):
    knn = KNeighborsClassifier(n_neighbors=k, **kwargs)
    knn.fit(X_train, y_train)
    return knn

def accuracy_vs_k(X_train, y_train, X_test, y_test, k_range=range(1,12), **kwargs):
    accs = []
    for k in k_range:
        m = KNeighborsClassifier(n_neighbors=k, **kwargs)
        m.fit(X_train, y_train)
        accs.append(m.score(X_test, y_test))
    return list(k_range), accs

def plot_k_accuracy(k_vals, accs):
    plt.figure(figsize=(7,4))
    plt.plot(k_vals, accs, marker='o')
    plt.title("Accuracy vs k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.xticks(k_vals)
    plt.grid(True)
    plt.show()

def confusion_and_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['Down','Up']))
    return cm

def plot_hist_vs_normal(series):
    mu = series.mean()
    sigma = series.std()
    x_vals = np.linspace(series.min(), series.max(), 200)
    pdf = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(- (x_vals - mu)**2 / (2 * sigma**2))
    plt.figure(figsize=(7,4))
    sns.histplot(series, bins=30, stat='density', color='lightgrey')
    plt.plot(x_vals, pdf, label='Normal PDF', linewidth=2)
    plt.title(f"{series.name}: Histogram vs Normal PDF")
    plt.xlabel(series.name)
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def accuracy_with_metrics(X_train, y_train, X_test, y_test, metric='minkowski', p=2):
    if metric == 'minkowski':
        model = KNeighborsClassifier(n_neighbors=3, metric=metric, p=p)
    else:
        model = KNeighborsClassifier(n_neighbors=3, metric=metric)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test), model

def plot_roc_curve(model, X_test, y_test):
    # Need probability scores
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:,1]
    else:
        # fallback: use distance-based scoring if no predict_proba
        probs = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    return roc_auc

def custom_knn_predict(X_train_np, y_train_np, X_test_np, k=3):
    preds = []
    # vectorized distance computation per test vector
    for x in X_test_np:
        d = np.sqrt(np.sum((X_train_np - x)**2, axis=1))
        idx = np.argpartition(d, k)[:k]
        vals = y_train_np[idx]
        preds.append(np.bincount(vals).argmax())
    return np.array(preds)

# ---- Main execution ----
FEATURES = ['LSTM_POLARITY','TEXTBLOB_POLARITY','VOLATILITY_10D','VOLATILITY_30D','LAST_PRICE']

try:
    df = load_and_clean(DATA_PATH, FEATURES, target_col='1_DAY_RETURN')
except FileNotFoundError as e:
    print(e)
    raise

print("Data loaded and cleaned. Rows:", len(df))
print("Feature types:\n", df[FEATURES].dtypes)
print("Class distribution:\n", df['Class'].value_counts())

# A1
X_all = df[FEATURES].reset_index(drop=True)
y_all = df['Class'].reset_index(drop=True)
intr0, intr1, inter = compute_intrainter_inter(X_all, y_all, sample_n=50)
plot_distance_distributions(intr0, intr1, inter)

# A2
mean_lstm, var_lstm = plot_hist_feature(df['LSTM_POLARITY'], bins=35, title="LSTM_POLARITY Distribution")
print(f"Mean (LSTM_POLARITY) = {mean_lstm:.6f}, Variance = {var_lstm:.6f}")

# A3
vecA = X_all.iloc[0].values
vecB = X_all.iloc[1].values
r_vals, mink_dists = minkowski_distance_between_two(vecA, vecB, r_values=range(1,11))
plot_minkowski(r_vals, mink_dists)
print("Minkowski distances r=1..10:", [f"{d:.6f}" for d in mink_dists])

# A4
X_train, X_test, y_train, y_test = split_data(X_all, y_all, test_size=0.3)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train class counts:\n", y_train.value_counts())
print("Test class counts:\n", y_test.value_counts())

# A5
knn3 = train_knn(X_train, y_train, k=3)
print("kNN trained (k=3).")

# A6
y_pred = knn3.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (k=3): {acc:.4f}")

# A7: show few predictions
pred_df = X_test.copy()
pred_df['Actual'] = y_test.values
pred_df['Predicted'] = y_pred
print("Sample predictions:\n", pred_df.head(5))

# A8: k sweep
k_vals, accs = accuracy_vs_k(X_train, y_train, X_test, y_test, k_range=range(1,12))
plot_k_accuracy(k_vals, accs)
for k, a in zip(k_vals, accs):
    print(f"k={k} -> accuracy={a:.4f}")

# A9
cm = confusion_and_report(y_test, y_pred)

# Optional O1
plot_hist_vs_normal(df['LSTM_POLARITY'])

# O2: different metrics for kNN (k=3)
metrics_to_try = [('euclidean', None), ('manhattan', None), ('chebyshev', None), ('minkowski', 3)]
for metric, p in metrics_to_try:
    if p:
        acc_metric, model_m = accuracy_with_metrics(X_train, y_train, X_test, y_test, metric='minkowski', p=p)
        label = f"minkowski(p={p})"
    else:
        acc_metric, model_m = accuracy_with_metrics(X_train, y_train, X_test, y_test, metric=metric)
        label = metric
    print(f"Metric {label}: accuracy={acc_metric:.4f}")

# O3: AUROC for k=3 model
roc_auc = plot_roc_curve(knn3, X_test, y_test)
print(f"k=3 ROC AUC = {roc_auc:.4f}")

# O4: compare sklearn kNN and custom implementation on a small subset (speed/sanity)
# Take small subset to keep custom routine fast
sub_train_frac = 0.05
sub_test_frac = 0.05
if len(X_train) >= 20 and len(X_test) >= 20:
    X_tr_sm, _, y_tr_sm, _ = sk_train_test_split(X_train, y_train, train_size=sub_train_frac, random_state=RANDOM_STATE, stratify=y_train)
    X_te_sm, _, y_te_sm, _ = sk_train_test_split(X_test, y_test, train_size=sub_test_frac, random_state=RANDOM_STATE, stratify=y_test)
    model_sklearn = KNeighborsClassifier(n_neighbors=3)
    model_sklearn.fit(X_tr_sm, y_tr_sm)
    y_sklearn = model_sklearn.predict(X_te_sm)
    acc_skl = accuracy_score(y_te_sm, y_sklearn)
    X_tr_np = X_tr_sm.values
    y_tr_np = y_tr_sm.values.astype(int)
    X_te_np = X_te_sm.values
    y_custom = custom_knn_predict(X_tr_np, y_tr_np, X_te_np, k=3)
    acc_custom = accuracy_score(y_te_sm, y_custom)
    print(f"Sklearn kNN (subset) accuracy: {acc_skl:.4f}")
    print(f"Custom kNN (subset) accuracy: {acc_custom:.4f}")
else:
    print("Dataset too small for subset comparison in O4.")

print("All done.")
