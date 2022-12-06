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

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



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


def runPCA(df: pd.DataFrame, YEARS: list, INCLUDE_POS, THREE_POS_FLAG,
           VARIANCE: float):

    mod_data = modifydataformodel(df, INCLUDE_POS, THREE_POS_FLAG)

    scaler = StandardScaler()
    X = scaler.fit_transform(mod_data)

    # Identify optimal PCA components through Elbow Plots beforehand.
    common.createElbowPlots(len(mod_data.columns), X, YEARS)

    # Reduce the data's dimensionality to a number of components that explain
    # a portion of the dataset's variance.
    X_transform = common.pcaTransform(X, VARIANCE)

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
