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


def som(df: pd.DataFrame, YEARS: list,
        INCLUDE_POS, THREE_POS_FLAG):
    print("---- Start SOM Clustering model ----")

    df_data = modifyDataForModel(df, INCLUDE_POS, THREE_POS_FLAG)
    x = common.normalizeData(df_data.to_numpy())
    print("Data for Model Modification: COMPLETE")

    nba_som = SOM(m=len(df['Pos'].unique()),
                  n=1,
                  dim=len(x[0]),
                  random_state=2)
    nba_som.fit(x, epochs=1)
    labels = nba_som.predict(x)

    # Ensure that all labels are corrected to be in range [0, 4]
    labels = labels
    df.loc[:, 'Cluster'] = labels

    #####################################
    # Evaluate the Model
    # 1) Output PIE concentration charts of the clusters
    # 2) Measure the Tightness of each cluster
    common.calcPositionConc(df, "SOM", YEARS, THREE_POS_FLAG)

    return common.reportClusterScores(df, YEARS, INCLUDE_POS)

