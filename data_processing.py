import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
solutions = pd.read_csv('D:\\code\\galaxy-zoo-the-galaxy-challenge\\training_solutions_rev1.csv', sep = ',')
solutions = solutions.loc[solutions["Class1.3"] < 0.5]
solutions = solutions.loc[:, ["GalaxyID", "Class1.1", "Class1.2"]]
solutions.columns = ["GalaxyID", "Smooth", "Class1.2"]
solutions.loc[solutions.Smooth >  0.5, "Smooth"] = 1
solutions.loc[solutions.Smooth <= 0.5, "Smooth"] = 0
solutions = solutions.loc[:, ["GalaxyID", "Smooth"]]
solutions = solutions.astype({"GalaxyID" : int, "Smooth" : int})
solutions.to_csv("D:\\code\\galaxy-zoo-the-galaxy-challenge\\solutions.csv", sep = ",", encoding = "utf-8")
def cropSave(imgID):
    img = Image.open("D:\\code\\galaxy-zoo-the-galaxy-challenge\\images_training_rev1\\images_training_rev1\\" + str(imgID) + ".jpg")
    img = img.resize(box=(112, 112, 312, 312), size=(128,128))
    img.save("D:\\code\\galaxy-zoo-the-galaxy-challenge\\images_modified\\" + str(imgID) + ".jpg", "jpeg")

for imgID in solutions["GalaxyID"]:
    cropSave(imgID)