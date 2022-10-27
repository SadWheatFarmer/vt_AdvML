################################################################################
#   File:   dataPrepration.py
#   Author: John Smutny
#   Course: ECE-5424: Advanced Machine Learning
#   Date:   10/19/2022
#   Description:
#       Analyze and make changes to the dataset used in the NBA Position
#       research paper. Dataset provided by Omri Goldstein.
#       https://www.kaggle.com/datasets/drgilermo/nba-players-stats?select=Seasons_Stats.csv
#
## Control flags and constants
################################################################################

OUTPUT_FILES = True

import pandas as pd
import numpy as np

##############################

def modifyData(df: pd.DataFrame, YEARS: list, REQ_GAMES, REQ_MIN) -> \
        pd.DataFrame:
    '''
        :param df:
        :return:

        Edit the dataset in the following ways
        1) Remove features not needed for model
        2) Apply filters
            Player must have played in specific years: years = { firstYear, SecondYear }
            Player must have played in at least x games.
            Player must have played at least y minutes per game played.
        '''

    ##########################
    # Feature Cleanup

    # Add a name to the used ID feature to use as the dataframe index
    df = df.rename(columns={'Unnamed: 0': "ID"})

    # Remove Features
    REMOVE_FEATURES = ['blanl', 'blank2']
    df = df.drop(columns=REMOVE_FEATURES)

    # Remove all player names of 'nan'
    df = df[~pd.isna(df['Player'])]

    # Set all missing GS values to 0. (this is not a required field)
    df['GS'] = df['GS'].replace(np.nan, 0)

    # The number of NAN 3PA and NAN FTA equal the number of 0 3PA and FTA.
    # Therefore, replace all NAN values with zero to pervent a divide-by-zero
    df['3P%'] = df['3P%'].replace(np.nan, 0)
    df['FT%'] = df['FT%'].replace(np.nan, 0)

    ##########################
    # Cleanup of the Position feature

    # Eliminate multiple positions. Only take the first position before a '-'.
    for id in range(len(df['Pos'])):
        pos = df['Pos'].iloc[id]
        dash_position = pos.find('-')
        if dash_position == -1:
            continue
        elif dash_position == 1:
            df['Pos'].iloc[id] = pos[:1]
        elif dash_position == 2:
            df['Pos'].iloc[id] = pos[:2]

    # One-Hot Encode the 'Pos' feature
    df_oneHot_pos = pd.get_dummies(df['Pos'], prefix='Pos')
    df_oneHot_pos['ID'] = df['ID']

    df = pd.merge(df, df_oneHot_pos, how='left', on='ID')
    df = df.drop(columns='Pos')


    ##########################
    # Player Filters

    # 1) Year filter
    df = df[(df['Year'] >= YEARS[0]) & (df['Year'] <= YEARS[1])]
    # 2) Game filter
    df = df[(df['G'] >= REQ_GAMES)]
    # 3) Time filter
    df = df[(df['MP'] >= REQ_GAMES * REQ_MIN)]


    return df


def combineData(df_player: pd.DataFrame, df_stats: pd.DataFrame) -> \
        pd.DataFrame:

    # Drop some of the player features
    RETAINED_FEATURES = ['Player', 'height', 'weight']
    df_player = df_player[RETAINED_FEATURES]

    df_stats = df_stats.join(df_player.set_index('Player'), on='Player')

    df_stats.insert(3, 'height', df_stats.pop('height'))
    df_stats.insert(4, 'weight', df_stats.pop('weight'))
    df_stats.insert(5, 'Age', df_stats.pop('Age'))

    return df_stats


##############################
##############################
##############################
# Load dataset

PLAYER_PATH = "../data/Players.csv"
DATA_PATH = "../data/Seasons_Stats.csv"

df_players = pd.read_csv(PLAYER_PATH)
df_stats = pd.read_csv(DATA_PATH)

df_data = combineData(df_players, df_stats)

##############################
# Output an initial Data Quality Report on the RAW data

from vt_ApplicationsOfML.Libraries.DataExploration.DataQualityReport import DataQualityReport

OUTPUT_PATH = "../data/dqr_ALL_Season_Stats.csv"
NON_NUMERIC_COLUMNS = ['Unnamed: 0', 'Player', 'Tm', 'Pos', 'blanl', 'blank2']

report_all = DataQualityReport()
report_all.quickDQR(df_data, df_data.columns, NON_NUMERIC_COLUMNS)

if OUTPUT_FILES:
    report_all.to_csv(OUTPUT_PATH)

##############################
# Pre-process the data

YEARS = [1950, 2009]
REQ_GAMES = 20
REQ_MIN = 10
df_data = modifyData(df_data, YEARS, REQ_GAMES, REQ_MIN)
print("Data Modification: COMPLETE")


##############################
# Output a Data Quality Report for pre-processed data

OUTPUT_PATH_DQR = "../data/dqr_FILTERED_Season_Stats_{}-{}.csv".format(YEARS[0], YEARS[1])
OUTPUT_PATH_DATA = "../data/Season_Stats_{}-{}.csv".format(YEARS[0], YEARS[1])
NON_NUMERIC_COLUMNS = ['Player', 'Tm', 'Pos']

report = DataQualityReport()
report.quickDQR(df_data, df_data.columns, NON_NUMERIC_COLUMNS)

if OUTPUT_FILES:
    report.to_csv(OUTPUT_PATH_DQR)
    df_data.to_csv(OUTPUT_PATH_DATA)

'''
NOTES - Filtered players data
1) Almost all of the features have filled in data. 'n_missing' and 'n_zero' 
are pretty low for nearly all features. Unlike the data for ALL years.
2) 'Pos' has a cardinality of 16. That has to change.
3) ALL data has 24691, and 2000-2009 has 4242 players.
4) Some data features like BPM and BLK started in 1973-74
5) Some features such as OBPM and DBPM can be negative. This is ok.
'''