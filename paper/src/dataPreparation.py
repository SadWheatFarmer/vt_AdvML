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
#   Input:
#       1) Season_Stats.csv - Dataset from Basketball-Reference.com 's
#       'Total' and 'Advanced' data.
#       2) Players.csv - Dataset from Kaggle (same location as INPUT 1)) that
#       contains the heights and weights of all players in INPUT 1.
#   Output:
#       1) Season_Stats_<year1>_<year2>.csv - A csv file containing a
#       FILTERED set of NBA players statistics for ALL years inbetween two
#       year inputs.
#       2) dqr_ALL_Season_Stats.csv - a DataQualityReport for every
#       provided season of the loaded statistics before entry filtering.
#       3) Season_Stats_<year1>-<year2>.csv - statistics of NBA Players
#       between the specified years after filtering.
#       4) dqr_FILTERED_Season_Stats_<year1>-<year2>.csv - DataQualityReport
#       of data entries after filtering and inbetween the specified years.

## Control flags and constants
# OUTPUT_FILES - Boolean. Decide if the script should create new files.
# PLAYER_PATH - File path to a dataset with player height and weight
# DATA_PATH - File path to a dataset with player statistics
# NON_NUMERIC_COLUMNS - List from DATA_PATH of features that are not Numeric.
# REQ_GAMES - Numeric. Filter to remove players that don't play enough games
#               in a season.
# REG_MIN - Numeric. Filter to remove players that don't play enough
#               'minutes per game' in a season.

################################################################################

OUTPUT_FILES = True

import pandas as pd
import numpy as np

##############################

def removeDuplicates(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Look through data for identically named players in the same year and
    average their entries statistics so that there is one entry per year per
    player.
    There could be multiple entries for one player if that player switches
    teams over the course of the same year.

    :param df: Input dataframe of the overall data
    :return: Pandas Dataframe with combined data entries
    '''

    NON_COLUMNS = ['ID', 'Year', 'Player', 'height', 'weight', 'Age', 'Pos', 'Tm']
    SUM_COLUMNS = ['G', 'GS', 'MP',  'OWS', 'DWS', 'WS', 'ORB', 'DRB', 'TRB',
                   'FG', 'FGA', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '3P',
                   '3PA', '2P', '2PA', 'FT', 'FTA']
    AVG_COLUMNS = ['PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                   'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'WS/48', 'OBPM',
                   'DBPM', 'BPM', 'VORP', 'FG%', '3P%', '2P%', 'eFG%', 'FT%']

    # loop through all present years. Adjustments are continued in each year.
    for year in df['Year'].unique():
        df_oneYear = df[df['Year'] == year]
        for name in df_oneYear['Player'].unique():

            # Sum data entries if a player's name occurs more than once in a
            # season.
            if df_oneYear['Player'].value_counts()[name] > 1:
                indicies = df_oneYear[df_oneYear['Player'] == name].index
                refRow = df[df['ID'] == indicies[0]]

                # For each feature, ether SUM or AVG all the data entries.
                for feature in df.columns:
                    # Loop through each found player entry.
                    for i in indicies[1:]:
                        if feature in SUM_COLUMNS:
                            refRow[feature] = refRow[feature].iat[0] + \
                                              df_oneYear[df_oneYear['ID'] ==
                                                         i][feature].iat[0]
                        elif feature in AVG_COLUMNS:
                            print("do something")
                        else:
                            break
            # If there is only one player data entry, do not do anything.
            else:
                continue




def modifyData(df: pd.DataFrame, YEARS: list, REQ_GAMES, REQ_MIN) -> \
        pd.DataFrame:
    '''
        :param df:
        :return: pd.DataFrame:

        Edit the dataset in the following ways
        1) Remove features not needed for model
        2) Consolidate players with multiple entries in a single season.
        3) Apply filters
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

    ##########################

    df = removeDuplicates(df)

    ##########################
    # Remove nan features if over a criteria
    NAN_LIMIT = 0.3
    for col in df.columns:
        nanCount = df[col].isnull().sum()
        if nanCount/len(df[col]) > NAN_LIMIT and\
                col not in ['GS', '3P', '3PA', '3P%', 'FT%']:
            print("Delete column {} with "
                  "{:.2f}% nan values".format(col, nanCount/len(df[col])*100))
            df = df.drop([col], axis=1)
        elif nanCount:
            df[col] = df[col].replace(np.nan, df[col].median())
        else:
            continue


    ##########################
    # Cleanup of the Position feature

    # Eliminate multiple positions. Only take the first position before a '-'.
    # TODO - Decide if to do 3 pos {G, F, C} or 5 {PG, SG, SF, PF, C}
    #        I think it will be based on the year. Check cardinality of
    #        'position' feature. IDEA: Always convert down.
    #        If over 5, then convert to 3.
    #        If 4, then convert to 3
    for id in df['ID']:
        pos = df.loc[id, 'Pos']
        dash_position = pos.find('-')
        if dash_position == -1:
            continue
        elif dash_position == 1:
            df.loc[id, 'Pos'] = pos[:1]
        elif dash_position == 2:
            df.loc[id, 'Pos'] = pos[:2]

    # One-Hot Encode the 'Pos' feature
    df_oneHot_pos = pd.get_dummies(df['Pos'], prefix='Pos')
    df_oneHot_pos['ID'] = df['ID']

    df = pd.merge(df, df_oneHot_pos, how='left', on='ID')


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
# Load datasets

PLAYER_PATH = "../data/Players.csv"
#DATA_PATH = "../data/Seasons_Stats.csv" #1950-2017
DATA_PATH = "../data/Seasons_Stats_1950_2022.csv" #1950-2022

df_players = pd.read_csv(PLAYER_PATH)
df_stats = pd.read_csv(DATA_PATH)

# Add specific features from 'PLAYER_PATH' to the 'DATA_PATH' dataset
df_data = combineData(df_players, df_stats)

##############################
# Output an initial Data Quality Report on the RAW data

from vt_AdvML.paper.lib.DataQualityReport import DataQualityReport

OUTPUT_PATH = "../data/dqr_ALL_Season_Stats.csv"
NON_NUMERIC_COLUMNS = ['Unnamed: 0', 'Player', 'Tm', 'Pos', 'blanl', 'blank2']

report_all = DataQualityReport()
report_all.quickDQR(df_data, df_data.columns, NON_NUMERIC_COLUMNS)

if OUTPUT_FILES:
    report_all.to_csv(OUTPUT_PATH)

##############################
# Pre-process the data

YEARS = [[1971, 1980],
         [1981, 1990],
         [1991, 2000],
         [2001, 2010],
         [2011, 2020]]
REQ_GAMES = 20
REQ_MIN = 10

for YEAR in YEARS:
    df_year = df_data[(df_data['Year'] >= YEAR[0]) & (df_data['Year'] <= YEAR[1])]
    report_decade = DataQualityReport()
    report_decade.quickDQR(df_year, df_year.columns, NON_NUMERIC_COLUMNS)

    if OUTPUT_FILES:
        OUTPUT_PATH_DQR = "../data/dqr_Season_Stats_{}-{}.csv".format(YEAR[0], YEAR[1])
        report_decade.to_csv(OUTPUT_PATH_DQR)

    df_year = modifyData(df_year, YEAR, REQ_GAMES, REQ_MIN)
    print("Data Modification: COMPLETE")


    ##############################
    # Output a Data Quality Report for pre-processed data

    OUTPUT_PATH_DQR = "../data/dqr_FILTERED_Season_Stats_{}-{}.csv".format(YEAR[0], YEAR[1])
    OUTPUT_PATH_DATA = "../data/Season_Stats_{}-{}.csv".format(YEAR[0], YEAR[1])
    NON_NUMERIC_COLUMNS = ['Player', 'Tm', 'Pos']

    report = DataQualityReport()
    report.quickDQR(df_year, df_year.columns, NON_NUMERIC_COLUMNS)

    if OUTPUT_FILES:
        report.to_csv(OUTPUT_PATH_DQR)
        df_year.to_csv(OUTPUT_PATH_DATA)

'''
NOTES - Filtered players data
1) Almost all of the features have filled in data. 'n_missing' and 'n_zero' 
are pretty low for nearly all features. Unlike the data for ALL years.
2) 'Pos' has a cardinality of 16. That has to change.
3) ALL data has 24691, and 2000-2009 has 4242 players.
4) Some data features like BPM and BLK started in 1973-74
5) Some features such as OBPM and DBPM can be negative. This is ok.
'''