import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import minkowski

# Load CSV files from the current folder
tweets_df = pd.read_csv('stock_tweets.csv')
finance_df = pd.read_csv('stock_yfinance_data.csv')

# Check and clean tweet column names
print("Tweets columns before cleaning:", tweets_df.columns.tolist())
tweets_df.columns = tweets_df.columns.str.strip()  # Remove leading/trailing spaces

# Rename sentiment column if needed
if 'sentiment' in tweets_df.columns:
    tweets_df.rename(columns={'sentiment': 'Sentiment'}, inplace=True)

# Convert date columns to datetime and remove timezone (fixes merge error)
tweets_df['Date'] = pd.to_datetime(tweets_df['Date']).dt.tz_localize(None)
finance_df['Date'] = pd.to_datetime(finance_df['Date']).dt.tz_localize(None)

# Merge datasets on Date
df = pd.merge(tweets_df, finance_df, on='Date')

# Filter only Positive and Negative sentiment classes
df = df[df['Sentiment'].isin(['Positive', 'Negative'])]

# Encode sentiment labels as 0 and 1
df['Sentiment'] = LabelEncoder().fit_transform(df['Sentiment'])

# Convert tweet text into numerical vectors using TF-IDF
tfidf = TfidfVectorizer(max_features=100)
text_feats = tfidf.fit_transform(df['Tweet']).toarray()

# Get numeric stock features
numeric_feats = df[['Open', 'Close', 'High', 'Low']].fillna(0).values

# Combine text and stock features
X = np.hstack((text_feats, numeric_feats))
y = df['Sentiment'].values

# ----------------------- A1 -----------------------
class0 = X[y == 0]
class1 = X[y == 1]

centroid0 = class0.mean(axis=0)
centroid1 = class1.mean(axis=0)

spread0 = class0.std(axis=0)
spread1 = class1.std(axis=0)

inter_class_distance = np.linalg.norm(centroid0 - centroid1)

print("A1: Spread Class 0:", spread0.mean())
print("A1: Spread Class 1:", spread1.mean())
print("A1: Inter-class distance:", inter_class_distance)

# ----------------------- A2 -----------------------
feature = df['Open'].fillna(0).values
plt.hist(feature, bins=20)
plt.title('A2: Histogram of Open Price')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

print("A2: Mean of Open:", np.mean(feature))
print("A2: Variance of Open:", np.var(feature))

# ----------------------- A3 -----------------------
vec1 = X[0]
vec2 = X[1]
distances = [minkowski(vec1, vec2, p=r) for r in range(1, 11)]
plt.plot(range(1, 11), distances)
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.title('A3: Minkowski Distance vs r')
plt.grid(True)
plt.show()

# ----------------------- A4 -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ----------------------- A5 -----------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ----------------------- A6 -----------------------
acc = knn.score(X_test, y_test)
print("A6: Accuracy (k=3):", acc)

# ----------------------- A7 -----------------------
preds = knn.predict(X_test)
print("A7: Predictions (first 10):", preds[:10])

# ----------------------- A8 -----------------------
accuracies = []
for k in range(1, 12):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies.append(acc)

plt.plot(range(1, 12), accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('A8: Accuracy vs k')
plt.grid(True)
plt.show()

# ----------------------- A9 -----------------------
conf_matrix = confusion_matrix(y_test, preds)
print("A9: Confusion Matrix:\n", conf_matrix)

report = classification_report(y_test, preds)
print("A9: Classification Report:\n", report)
