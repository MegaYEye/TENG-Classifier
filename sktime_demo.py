from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head,load_japanese_vowels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_japanese_vowels(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.head())
classifier = TimeSeriesForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))