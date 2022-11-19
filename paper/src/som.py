################################################################################
#   File:   som.py
#   Author: John Smutny
#   Course: ECE-5424: Advanced Machine Learning
#   Date:   10/30/2022
#   Description:
#       Self-Organizing Map model to analyze NBA positions. Model provided by
#       Ripley Smith and their team.
#
#
#   Reference
#       Sklearn-somhttps://pypi.org/project/sklearn-som/#:~:text=sklearn%2Dsom%20is%20a%20minimalist,data%20and%20performing%20dimensionality%20reduction.
#       Source Code: https://github.com/rileypsmith/sklearn-som
################################################################################

import pandas as pd
from sklearn_som.som import SOM

import lib.modelCommon as common


def modifyDataForModel(df: pd.DataFrame,
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


def som(df: pd.DataFrame, YEARS: list,
        INCLUDE_POS, THREE_POS_FLAG) -> bool:
    print("---- Start SOM Clustering model ----")

    df_data = modifyDataForModel(df, INCLUDE_POS, THREE_POS_FLAG)
    x = common.normalizeData(df_data.to_numpy())
    print("Data for Model Modification: COMPLETE")

    nba_som = SOM(m=len(df['Pos'].unique()), n=1, dim=len(x[0]))
    nba_som.fit(x, epochs=1)
    labels = nba_som.predict(x)

    # Ensure that all labels are corrected to be in range [1, 5]
    labels = labels + 1
    df['Cluster'] = labels

    # Isolate positions per player in the clusters
    # df_cluster = df[['ID', 'Year', 'Player', 'Pos', 'Cluster']]
    # df_cluster.to_csv("../output/MODEL_Labels_SOM_{}-{}.csv".format(
    #                     years[0],
    #                     years[1]),
    #                     index=False)

    common.calcPositionConc(df, "SOM", YEARS)
    common.reportClusterScores(df, INCLUDE_POS)

    return df_data

