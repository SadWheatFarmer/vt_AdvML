'''
File:   dataPrepration.py
Author: John Smutny
Course: ECE-5424: Advanced Machine Learning
Date:   11/19/2022
Description:
    Analyze and make changes to the dataset used in the NBA Position
    research paper. Dataset provided by Omri Goldstein.
    https://www.kaggle.com/datasets/drgilermo/nba-players-stats?select=Seasons_Stats.csv

Input:
    1) Season_Stats.csv - Dataset from Basketball-Reference.com 's
    'Total' and 'Advanced' data.
    2) Players.csv - Dataset from Kaggle (same location as INPUT 1)) that
    contains the heights and weights of all players in INPUT 1.
Output:
    1) Season_Stats_<year1>_<year2>.csv - A csv file containing a
    FILTERED set of NBA players statistics for ALL years inbetween two
    year inputs.
    ** Only created if the LOAD_MODEL_DATA flag is set to false
'''

import pandas as pd
import numpy as np
from lib.DataQualityReport import DataQualityReport


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

    STATIC_COLUMNS = ['ID', 'Year', 'Player', 'height', 'weight', 'Age', 'Pos', 'Tm']
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

                # For each feature, ether SUM or AVG all the data entries.
                for feature in df.columns:
                    # Loop through each found player entry. Make a list of
                    # values.
                    vals = []
                    if feature in STATIC_COLUMNS:
                        continue
                    else:
                        for i in indicies[:]:
                            vals.append(
                                df_oneYear[df_oneYear['ID'] == i][feature].iat[0])

                        if feature in SUM_COLUMNS:
                            val = sum(vals)
                        elif feature in AVG_COLUMNS:
                            val = sum(vals) / len(vals)
                        else:
                            break

                        # Sanity Check. Replace the value if result doesn't
                        # make any sense.
                        # 1) Apply the collective value
                        # 2) or do not make any modification due to data
                        # weirdness.
                        replaceValue = False
                        if feature in ['G', 'GS']:
                            if val <= 82:
                                replaceValue = True
                            else:
                                replaceValue = True
                                val = 82
                        else:
                            replaceValue = True

                        if replaceValue:
                            df.at[indicies[0], feature] = val

                # Delete non-first entries
                df = df.drop(indicies[1:])


            # If there is only one player data entry, do not do anything.
            else:
                continue

    print("**** Data Modification: removeDuplicates - COMPLETE")

    return df


def cleanPositionFeature(df: pd.DataFrame, THREE_POSITIONS_FLAG) -> pd.DataFrame:
    '''
    1) Only let a player have one position.
           Only take the first position before a '-' character.

    2) Limit the cardinality of the 'position' feature based on the input
     THREE_POSITIONS_FLAG.
        if True, then convert down all specialized Guard 'G' and Forward 'F'
            positions into their simpler equivalents.
            - numPos3 = ['G', 'F', 'C']
        if False, then ensure all positions are one of the following:
            - numPos5 = ['PG', 'SG', 'SF', 'PF', 'C']

    :param df: dataframe with all statistical data
    :param THREE_POSITIONS_FLAG: boolean defining how many position options.
    :return: Dataframe with updated 'Position' feature.
    '''

    for id in df['ID']:
        # 1) In each entry, only use the first position listed if a player is
        # listed with multiple positions. Ex: PG-SG, F-C, G-F, etc
        pos = df.loc[id, 'Pos']
        dash_position = pos.find('-')

        # If found dash char '-' is not found (-1) then do nothing
        if dash_position == 1:
            pos = pos[:1]
        elif dash_position == 2:
            pos = pos[:2]

        # 2) Only allow 3 or 5 cardinality for the 'position' feature
        if THREE_POSITIONS_FLAG:
            if pos == 'PG' or pos == 'SG':
                pos = 'G'
            elif pos == 'SF' or pos == 'PF':
                pos = 'F'
        else:
            if pos == 'G':
                pos = 'SG'
            elif pos == 'F':
                pos = 'SF'

        # Set the resulting position categorical.
        df.loc[id, 'Pos'] = pos

    return df

def modifyData(df: pd.DataFrame, YEARS: list,
               REQ_GAMES, REQ_MIN,
               THREE_POSITIONS_FLAG) -> \
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

    print("*** Data Modification {}-{}: START".format(YEARS[0], YEARS[1]))

    ##########################
    # Initial Player Filter

    # 1) Year filter
    df = df[(df['Year'] >= YEARS[0]) & (df['Year'] <= YEARS[1])]


    ##########################
    # Feature Cleanup

    # Add a name to the used ID feature to use as the dataframe index
    df = df.rename(columns={'Unnamed: 0': "ID"})

    # Remove features that could break the program execution
    REMOVE_FEATURES = ['blanl', 'blank2']
    df = df.drop(columns=REMOVE_FEATURES)

    # Remove all player names of 'nan'
    df = df[~pd.isna(df['Player'])]

    ##########################
    # Consolidate any entries that are listed more than once.
    df = removeDuplicates(df)

    ##########################
    # Remove nan features if over a criteria
    NAN_LIMIT = 0.3
    for col in df.columns:
        nanCount = df[col].isnull().sum()
        if nanCount/len(df[col]) > NAN_LIMIT and\
                col not in ['GS', '3P', '3PA', '3P%', 'FT%']:
            print("**** Delete column {} with "
                  "{:.2f}% nan values".format(col, nanCount/len(df[col])*100))
            df = df.drop([col], axis=1)
        elif nanCount:
            df[col] = df[col].replace(np.nan, df[col].median())
        else:
            continue
    print("**** Data Modification: Remove NaN Features - COMPLETE")

    ##########################
    # Ensure 'Position' feature has only 3 or 5 categories if included.
    df = cleanPositionFeature(df, THREE_POSITIONS_FLAG)

    # Add in One-Hot Encoding 'Pos' feature values.
    df_oneHot_pos = pd.get_dummies(df['Pos'], prefix='Pos')
    df_oneHot_pos['ID'] = df['ID']

    df = pd.merge(df, df_oneHot_pos, how='left', on='ID')

    ##########################
    # Specific Player Filters

    # 2) Game filter
    df = df[(df['G'] >= REQ_GAMES)]
    # 3) Time filter
    df = df[(df['MP'] >= REQ_GAMES * REQ_MIN)]

    print("*** Data Modification {}-{}: COMPLETE".format(YEARS[0], YEARS[1]))

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


def outputReferenceFiles(df_RAW: pd.DataFrame, df_MODEL: pd.DataFrame,
                         OUTPUT_PATH, YEARS_PAIRS,
                         NON_NUMERIC_COLUMNS):

    ##############################
    # Output an initial Data Quality Report on the RAW data
    OUTPUT_PATH_RAW = OUTPUT_PATH + "Season_Stats_dqr_RAW.csv"
    report_raw = DataQualityReport()
    report_raw.quickDQR(df_RAW, df_RAW.columns, NON_NUMERIC_COLUMNS)
    report_raw.to_csv(OUTPUT_PATH_RAW)

    ##############################
    # Output a .csv and a Data Quality Report for the data used in modeling
    OUTPUT_PATH_DQR = OUTPUT_PATH + "Season_Stats_dqr_MODEL_{}-{}.csv".format(
        YEARS_PAIRS[0][0], YEARS_PAIRS[len(YEARS_PAIRS)-1][1])
    report_raw = DataQualityReport()
    report_raw.quickDQR(df_MODEL, df_MODEL.columns, NON_NUMERIC_COLUMNS)
    report_raw.to_csv(OUTPUT_PATH_DQR)

    OUTPUT_PATH_MODEL = OUTPUT_PATH + "Season_Stats_MODEL_{}-{}.csv".format(
        YEARS_PAIRS[0][0], YEARS_PAIRS[len(YEARS_PAIRS)-1][1])
    df_MODEL.to_csv(OUTPUT_PATH_MODEL)

    ##############################
    # Output a DataQualityReport and dataset for each Year-Pair
    for YEAR in YEARS_PAIRS:
        df_RAW_year = df_RAW[(df_RAW['Year'] >= YEAR[0])
                          & (df_RAW['Year'] <= YEAR[1])]
        df_MODEL_year = df_RAW[(df_RAW['Year'] >= YEAR[0])
                             & (df_RAW['Year'] <= YEAR[1])]

        ##############################
        # Output a Data Quality Report for pre-processed data
        OUTPUT_PATH_DQR = OUTPUT_PATH + "Season_Stats_dqr_RAW_{}-{}.csv".format(
            YEAR[0], YEAR[1])
        report_decade = DataQualityReport()
        report_decade.quickDQR(df_RAW_year, df_RAW_year.columns,
                               NON_NUMERIC_COLUMNS)
        report_decade.to_csv(OUTPUT_PATH_DQR)

        ##############################
        # Output a Data Quality Report for processed data
        OUTPUT_PATH_DQR = OUTPUT_PATH + "Season_Stats_dqr_MODEL_{}-{}.csv".format(
            YEAR[0], YEAR[1])
        report = DataQualityReport()
        report.quickDQR(df_MODEL_year, df_MODEL_year.columns, NON_NUMERIC_COLUMNS)
        report.to_csv(OUTPUT_PATH_DQR)


##############################
##############################
##############################


def initialDataModification(PLAYER_PATH, DATA_PATH,
                            YEARS_PAIRS, REQ_GAMES, REQ_MIN,
                            THREE_POSITION_FLAG, NON_NUMERIC_COLUMNS,
                            OUTPUT_FILES_FLAG):

    # Load datasets
    df_players = pd.read_csv(PLAYER_PATH)
    df_stats = pd.read_csv(DATA_PATH)

    # Add specific features from 'PLAYER_PATH' to the 'DATA_PATH' dataset
    df_data = combineData(df_players, df_stats)

    # Process data using indicated constraints for ONLY the relevant years.
    YEAR_RANGE = [YEARS_PAIRS[0][0], YEARS_PAIRS[len(YEARS_PAIRS)-1][1]]
    df_model = modifyData(df_data, YEAR_RANGE, REQ_GAMES, REQ_MIN,
                          THREE_POSITION_FLAG)

    df_model.to_csv("../data/ref/Season_Stats_MODEL_{}-{}.csv".format(
                    YEAR_RANGE[0], YEAR_RANGE[1]),
                    index=False)

    # IF DESIRED, output various DQR and csv files about the data.
    if OUTPUT_FILES_FLAG:
        OUTPUT_PATH = "../data/ref/"
        outputReferenceFiles(df_data, df_model,
                             OUTPUT_PATH, YEARS_PAIRS, NON_NUMERIC_COLUMNS)

    return df_model



# '''
# NOTES - Filtered players data
# 1) Almost all of the features have filled in data. 'n_missing' and 'n_zero'
# are pretty low for nearly all features. Unlike the data for ALL years.
# 2) 'Pos' has a cardinality of 16. That has to change.
# 3) ALL data has 24691, and 2000-2009 has 4242 players.
# 4) Some data features like BPM and BLK started in 1973-74
# 5) Some features such as OBPM and DBPM can be negative. This is ok.
# '''
