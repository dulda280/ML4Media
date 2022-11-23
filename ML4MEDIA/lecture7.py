import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import feature_selection

if __name__ == '__main__':
    diabetusData = pd.read_csv('diabetes.csv')
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        # print(diabetusData)
        pass

    preg = diabetusData.iloc[:, 0]
    gluc = diabetusData.iloc[:, 1]
    bloodP = diabetusData.iloc[:, 2]
    skinthic = diabetusData.iloc[:, 3]
    insulin = diabetusData.iloc[:, 4]
    bmi = diabetusData.iloc[:, 5]
    diaBprediG = diabetusData.iloc[:, 6]
    age = diabetusData.iloc[:, 7]
    diaB = diabetusData.iloc[:, 8]

    x = []
    y = []

    for i in range(0, len(diaB)):
        x.append([preg[i], gluc[i], bloodP[i], skinthic[i], insulin[i], bmi[i], diaBprediG[i], age[i]])
        y.append(diaB[i])

    bestFeet = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=4)

    #### Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome ####
    featureNAmes = ['Preg', 'Gluc', 'BloodP', 'SkinThicc', 'Insluin', 'BMI', 'DiaBpediG', 'Age']
    bestFeets = bestFeet.fit_transform(x, y)
    names = bestFeet.get_feature_names_out(featureNAmes)

    print(names)
    #### Gluc, insulin, BMI, age ###
    print(bestFeets)
    plt.matshow(diabetusData.corr())
    correMAT = diabetusData.corr()
    print(correMAT)
    plt.show()
    glucCorr = correMAT.iloc[8, 1]
    insCorr = correMAT.iloc[8, 4]
    BMICorr = correMAT.iloc[8, 5]
    ageCorr = correMAT.iloc[8, 7]

    glucIns = correMAT.iloc[1, 4]

    data = diabetusData[['Glucose', 'Insulin', 'BMI', 'Age']]
    print(data.corr())
