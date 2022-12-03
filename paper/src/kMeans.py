'''
File:   kMeans.py
Author: Jason Cusati
Course: ECE-5424: Advanced Machine Learning
Date:   11/28/2022
Description:
    kMeans model to analyze NBA positions.
'''


import pandas as pd
import lib.modelCommon as common

import sklearn
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import seaborn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import collections


def modifyDataForModel(df: pd.DataFrame,
                       INCLUDE_POS_FLAG, THREE_POS_FLAG) -> pd.DataFrame:

    # Remove Features
    REMOVE_FEATURES = ['ID', 'Year', 'Player', 'Tm', 'Pos']

    # Also delete position features if they should not be used in modeling.
    if not INCLUDE_POS_FLAG:
        if THREE_POS_FLAG:
            REMOVE_FEATURES.extend(["Pos_G", "Pos_F", "Pos_C"])
        else:
            REMOVE_FEATURES.extend(["Pos_PG", 'Pos_SG',
                                    "Pos_SF", "Pos_PF",
                                    "Pos_C"])

    df = df.drop(columns=REMOVE_FEATURES)

    return df


def runKmeans(df: pd.DataFrame, YEARS: list, INCLUDE_POS, THREE_POS_FLAG) -> bool:

    print(df.head())
    mod_data = modifyDataForModel(df, INCLUDE_POS, THREE_POS_FLAG)

    print(mod_data.head())
    scaler = StandardScaler()
    X = scaler.fit_transform(mod_data)

    num_clusters = len(df['Pos'].unique())

    print(pd.DataFrame(X).head())
    kmeans = KMeans(n_clusters=num_clusters,
                    init='k-means++',
                    max_iter=300,
                    n_init=10,
                    random_state=0)
    pred_y = kmeans.fit_predict(X)

    print(X[:,0])
    print("Inertia:\t")
    print(kmeans.inertia_)
    labels = kmeans.predict(X)

    print ("Clusters (result of k-means)")
    print (collections.Counter(pred_y))

    print ("Ground truth")
    print (collections.Counter(df['Pos']))
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Pos'], X[:,1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    #plt.show()

    # Ensure that all labels are corrected to be in range [0, 4]
    df.loc[:, 'Cluster'] = pred_y

    common.calcPositionConc(df, "kMeans", YEARS, THREE_POS_FLAG)

    return common.reportClusterScores(df, YEARS, INCLUDE_POS)

