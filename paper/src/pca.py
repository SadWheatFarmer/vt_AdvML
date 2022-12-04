'''
File:   pca.py
Author: Jason Cusati
Course: ECE-5424: Advanced Machine Learning
Date:   11/28/2022
Description:
    kMeans model to analyze NBA positions.
'''

import pandas as pd
import lib.modelCommon as common
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


def modifydataformodel(df: pd.DataFrame,
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

def createElbowPlots(mod_data: pd.DataFrame, X, YEARS: list):

    pca = PCA(n_components=len(mod_data.columns))
    pca.fit(X)

    # Display the Elbow Plot explaining the optimal # of PCA components
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Explained Variance')
    plt.savefig('../model/Elbow_Plot_PCA-{}-{}.png'.format(YEARS[0],
                                                       YEARS[1],
                                                       dpi=100))


def pcaTransform(df: pd.DataFrame, VARIANCE: int) -> pd.DataFrame:
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # Perform PCA on transformed dataset by using components with a
    # percentage of the explained dataset variance.
    pca = PCA(n_components=VARIANCE)
    pca.fit(X)
    print(
        "explained variance ratio by Components: {:.2f}%"
            "\n\tComponent (0-100%): {}".format(
            sum(pca.explained_variance_ratio_*100),
            pca.explained_variance_ratio_*100)
    )

    X_transform = pca.transform(X)
    return X_transform

def runPCA(df: pd.DataFrame, YEARS: list, INCLUDE_POS, THREE_POS_FLAG,
           VARIANCE):

    mod_data = modifydataformodel(df, INCLUDE_POS, THREE_POS_FLAG)

    scaler = StandardScaler()
    X = scaler.fit_transform(mod_data)

    # Identify optimal PCA components through Elbow Plots beforehand.
    createElbowPlots(mod_data, X, YEARS)

    X_transform = pcaTransform(X, VARIANCE)

    plt.figure()
    colors = ["navy", "turquoise", "darkorange", "darkgreen", "maroon"]
    lw = 2

    y = df['Pos']
    target_names = df['Pos'].unique()
    # TODO - Try the best you can to order the positions in order.
    #  ['PG', 'SG', 'SF', 'PF', 'C']

    for color, i, target_name in zip(colors,
                                     ['PG', 'SG', 'SF', 'PF', 'C'],
                                     target_names):
        plt.scatter(  X_transform[y == i, 0],
                      X_transform[y == i, 1],
                      color=color,
                      alpha=0.8,
                      lw=lw,
                      label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of NBA dataset")
    plt.savefig("../model/SCATTER_{}_Season_Stats_{}-{}".format(
                "PCA",
                YEARS[0],
                YEARS[1]))
    #print("Transform: %s:" % str(X_transform))
    #print("Components: %s" % pca.components_)

    # Perform cluster modeling on the resulting PCA components
    num_clusters = len(df['Pos'].unique())
    kmeans = KMeans(n_clusters=num_clusters,
                    init='k-means++',
                    max_iter=300,
                    n_init=10,
                    random_state=0)
    pred_y = kmeans.fit_predict(X_transform)
    df.loc[:, 'Cluster'] = pred_y

    common.calcPositionConc(df, "PCA", YEARS, THREE_POS_FLAG)

    return common.reportClusterScores(df, YEARS, INCLUDE_POS)
