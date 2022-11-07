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

def modifyDataForModel(df: pd.DataFrame, pos) -> pd.DataFrame:

    # Add the extra id column. BUG
    df = df.drop(columns='Unnamed: 0')

    # Remove Features
    if pos:
        REMOVE_FEATURES = ['Player', 'Tm', 'Pos']
        print("Include player POSITION in the model.")
    else:
        REMOVE_FEATURES = ['Player', 'Tm', 'Pos',
                           "Pos_C", "Pos_F", "Pos_G", "Pos_PF",
                           "Pos_PG", "Pos_SF", 'Pos_SG']
        print("Remove player POSITION from the model.")

    df = df.drop(columns=REMOVE_FEATURES)

    return df

def som(df: pd.DataFrame, years: list) -> bool:
    print("---- Start SOM Clustering model ----")

    df_data = modifyDataForModel(df, True)
    x = normalizeData(df_data.to_numpy())
    print("Data for Model Modification: COMPLETE")

    nba_som = SOM(m=5, n=1, dim=len(x[0]))
    nba_som.fit(x, epochs=1)
    labels = nba_som.predict(x)

    df['Cluster'] = labels

    return True

