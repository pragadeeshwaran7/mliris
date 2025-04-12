import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

clf = RandomForestClassifier()
clf.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump({"model": clf, "class_names": class_names}, f)
