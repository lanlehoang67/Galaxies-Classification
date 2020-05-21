import numpy as np
import pandas as pd
from PIL import Image
import xgboost as xgb
from sklearn.metrics import accuracy_score
from joblib import dump
from joblib import load
solutions = pd.read_csv("D:\\code\\galaxy-zoo-the-galaxy-challenge\\solutions.csv")
solutions = solutions.truncate(after = 11999, axis = "rows")
solutions.shape
def getImage(imgID):
    return np.array(Image.open("D:\\code\\galaxy-zoo-the-galaxy-challenge\\images_modified\\" + str(imgID) + ".jpg")).flatten()
X_train = np.array([np.array(getImage(imgID)) for imgID in solutions.truncate(after=9599, axis='rows')["GalaxyID"]]) / 255
X_test  = np.array([np.array(getImage(imgID)) for imgID in solutions.truncate(before=9600, after=11999, axis='rows')["GalaxyID"]]) / 255
y_train = solutions.truncate(after=9599, axis='rows')
y_test  = solutions.truncate(before=9600, after=11999, axis='rows')
del solutions
y_train = y_train.loc[:, ["Smooth"]]
y_test = y_test.loc[:, ["Smooth"]]
model = xgb.XGBClassifier(
    silent=True,
    n_jobs=-1,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=1000,
    objective='binary:logistic',
    min_child_weight=3,
    subsample=0.7,
    colsample_bytree=0.9,
    reg_alpha=1,
)

model.fit(X_train, y_train.values.ravel())
# model = load('D:\\GalaxyClassification\\xgboost')
# y_pred = model.predict(X_test)

y_pred = np.round(y_pred)
# print("Model test accuracy is:" + str(accuracy_score(y_test, y_pred)*100) + "%")

dump(model, 'xgboost')

