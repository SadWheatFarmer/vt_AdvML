'''
File:   kMeans.py
Author: Jason Cusati
Course: ECE-5424: Advanced Machine Learning
Date:   11/28/2022
Description:
    kMeans model to analyze NBA positions.
'''

import pandas as pd
import paper.src.lib.modelCommon as common
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def modifydataformodel(df: pd.DataFrame,
                       INCLUDE_POS_FLAG, THREE_POS_FLAG) -> pd.DataFrame:
    # Remove Features
    REMOVE_FEATURES = ['ID', 'Player', 'Tm', 'Pos']

    # Also delete position features if they should not be used in modeling.
    if ~INCLUDE_POS_FLAG:
        if THREE_POS_FLAG:
            REMOVE_FEATURES.extend(["Pos_G", "Pos_F", "Pos_C"])
        else:
            REMOVE_FEATURES.extend(["Pos_PG", 'Pos_SG',
                                    "Pos_SF", "Pos_PF",
                                    "Pos_C"])

    df = df.drop(columns=REMOVE_FEATURES)

    return df


def runPCA(df: pd.DataFrame, YEARS: list, INCLUDE_POS, THREE_POS_FLAG) -> bool:
    mod_data = modifydataformodel(df, INCLUDE_POS, THREE_POS_FLAG)
    scaler = StandardScaler()
    X = scaler.fit_transform(mod_data)
    num_clusters = len(df['Pos'].unique())
    pca = PCA(n_components=num_clusters)
    pca.fit(X)
    X_transform = pca.transform(X)

    print(
        "explained variance ratio: %s"
        % str(pca.explained_variance_ratio_)
    )

    plt.figure()
    colors = ["navy", "turquoise", "darkorange", "darkgreen", "maroon"]
    lw = 2

    y = df['Pos']
    target_names = df['Pos'].unique()

    for color, i, target_name in zip(colors, ['C', 'SF', 'PG', 'SG', 'PF'], target_names):
        plt.scatter(
            X_transform[y == i, 0], X_transform[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of NBA dataset")
    plt.savefig("../model/SCATTER_{}_Season_Stats_{}-{}".format(
                "PCA",
                YEARS[0],
                YEARS[1]))
    print("Transform: %s:"
          % str(X_transform))
    print(
        "Components: %s"
        % pca.components_)

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X_transform)

    df['Cluster'] = pred_y

    common.calcPositionConc(df, "Hierarchy", YEARS, THREE_POS_FLAG)

    return common.reportClusterScores(df, YEARS, INCLUDE_POS)
