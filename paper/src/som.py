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
## Control flags and constants
################################################################################


import numpy as np
import pandas as pd
from sklearn_som.som import SOM
from sklearn.preprocessing import StandardScaler, normalize

def normalizeData(np_array) -> np.array:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np_array.tolist())

    x_normalized = normalize(x_scaled)

    return x_normalized

def modifyDataForModel(df: pd.DataFrame, INCLUDE_POS) -> pd.DataFrame:

    # Add the extra id column. BUG
    df = df.drop(columns='Unnamed: 0')

    # Remove Features
    REMOVE_FEATURES = ['ID', 'Player', 'Tm', 'Pos']
    if INCLUDE_POS:
        print("Include ONE-HOT ENCODED player POSITION in the model.")
    else:
        if len(df[df['Pos'] == 'G']) > 0:
            REMOVE_FEATURES.extend(["Pos_C", "Pos_F", "Pos_G"])
        else:
            REMOVE_FEATURES.extend(["Pos_PG", 'Pos_SG',
                                    "Pos_SF", "Pos_PF",
                                    "Pos_C"])
        print("Remove ONE-HOT ENCODED player POSITION from the model.")

    df = df.drop(columns=REMOVE_FEATURES)

    return df

def som(df: pd.DataFrame, years: list, includePos) -> bool:
    print("---- Start SOM Clustering model ----")

    df_data = modifyDataForModel(df, includePos)
    x = normalizeData(df_data.to_numpy())
    print("Data for Model Modification: COMPLETE")

    nba_som = SOM(m=len(df['Pos'].unique()), n=1, dim=len(x[0]))
    nba_som.fit(x, epochs=1)
    labels = nba_som.predict(x)

    # Ensure that all labels are corrected to be in range [1, 5]
    labels = labels + 1
    df['Cluster'] = labels

    return True

