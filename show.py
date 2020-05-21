from numpy import loadtxt
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
model = load('D:\\GalaxyClassification\\xgboost')
results = cross_val_score(model)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))