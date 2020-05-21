import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from PIL import Image
import xgboost as xgb
from joblib import load

import matplotlib.pyplot as plt
from random import randint
data = pd.read_csv('D:\\code\\galaxy-zoo-the-galaxy-challenge\\training_solutions_rev1.csv')

correctPredictions = 0
total = 0
model = load('D:\\GalaxyClassification\\xgboost')

while(True):
    i = randint(20000, 60000)
    if data.at[i, "Class1.3"] <= 0.5:
        break

plt.figure(figsize=(9, 9))
imgId = data.at[i, "GalaxyID"]
img = Image.open("D:\\code\\galaxy-zoo-the-galaxy-challenge\\images_training_rev1\\images_training_rev1\\" + str(imgId) + ".jpg")
plt.imshow(img)

img = Image.open("D:\\code\\galaxy-zoo-the-galaxy-challenge\\images_modified\\" + str(imgId) + ".jpg")
img = np.array([np.array(img).flatten()])
img = img/255

prediction = model.predict(np.array([np.array(img).flatten()]))[0]

correct = round(data.at[i, "Class1.1"])

print("Image ID: " + str(imgId))

if prediction == 0:
    print("Model says it is SPIRAL")
    print(str(int(data.at[i, "Class1.2"] * 100)) + "% of people think so")
else:
    print("Model says it is ELLIPTICAL")
    print(str(int(data.at[i, "Class1.1"] * 100)) + "% of people think so")

print("CORRECT!" if prediction == correct else "WRONG!")
if prediction == correct:
    correctPredictions += 1
total += 1

print("Correct predictions: " + str(int(correctPredictions / total * 100)) + "%")
print("Predictions made: " + str(total))
