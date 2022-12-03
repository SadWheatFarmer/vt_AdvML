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

def combineDuplicates(df: pd.DataFrame, df_matches: pd.DataFrame) -> \
        pd.DataFrame:
    STATIC_COLUMNS = ['ID', 'Year', 'Player', 'height', 'weight', 'Age', 'Pos', 'Tm']
    SUM_COLUMNS = ['G', 'GS', 'MP',  'OWS', 'DWS', 'WS', 'ORB', 'DRB', 'TRB',
                   'FG', 'FGA', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '3P',
                   '3PA', '2P', '2PA', 'FT', 'FTA']
    AVG_COLUMNS = ['PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                   'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'WS/48', 'OBPM',
                   'DBPM', 'BPM', 'VORP', 'FG%', '3P%', '2P%', 'eFG%', 'FT%']

    indicies = df_matches.index

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
                    df_matches[df_matches['ID'] == i][feature].iat[0])

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

            if feature in ['G', 'GS']:
                # Cap the number of games to the max in a regular season
                if val > 82:
                    val = 82

            df.at[indicies[0], feature] = val

    # Delete non-first entries
    df = df.drop(indicies[1:])

    return df


def removeDuplicates2(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Look through data for identically named players in the same year and
    average their entries statistics so that there is one entry per year per
    player.
    There could be multiple entries for one player if that player switches
    teams over the course of the same year.

    :param df: Input dataframe of the overall data
    :return: Pandas Dataframe with combined data entries
    '''

    # Count of entries effected by this process
    count = 0

    # loop through all present years. Adjustments are continued in each year.
    for year in df['Year'].unique():
        df_oneYear = df[df['Year'] == year]

        # Isolate every unique player name. Does not guarantee a unique player.
        for name in df_oneYear['Player'].unique():

            # Only attempt to combine entries if a player entry is unique
            if df_oneYear['Player'].value_counts()[name] == 1:
                continue
            # Mark players as unique individuals by matching name, age, and pos
            else:
                df_nameMatch = df_oneYear[df_oneYear['Player'] == name]

                # Checks to see if there are more than one player that has
                # the same name but multiple entries.
                if (len(df_nameMatch['Age'].unique()) == 1 and
                        len(df_nameMatch['weight'].unique()) == 1 and
                        len(df_nameMatch['Pos'].unique()) == 1):
                    df = combineDuplicates(df, df_nameMatch)
                    count = count + len(df_nameMatch.index[1:])

                # Continue to refine the player. Same name but different markers
                else:
                    for age in df_nameMatch['Age'].unique():
                        df_ageMatch = df_nameMatch[df_nameMatch['Age'] == age]

                        # Matched player hase the same name, age, and weight.
                        # Therefore, the player(s) is only one person.
                        if len(df_ageMatch['weight'].unique()) == 1:
                            df = combineDuplicates(df, df_ageMatch)
                            count = count + len(df_ageMatch.index[1:])

                        # Matched player(s) have the same name and age but
                        # have a different weight. Therefore, they are different
                        # people.
                        else:
                            for weight in df_ageMatch['weight'].unique():
                                df_wMatch= df_ageMatch['weight'] == weight

                                if len(df_wMatch['weight'].unique()) == 1:
                                    df = combineDuplicates(df, df_wMatch)

                                # Weird circumstance. Combine players
                                else:
                                    print("****** ::replaceDuplicates() - "
                                          "Weird case {}".format(name))
                                    df.loc[:, df_wMatch.index] = \
                                        combineDuplicates(
                                            df_wMatch)

                                count = count + len(df_ageMatch.index[1:])

                                # TODO - Further refinement is the 1971 case
                                #  (Len Chappell - ID3105) where a player has
                                #  same name, age, weight but played a
                                #  different position on another team.
                                #  FIX: Compare the number of games played (
                                #  'G'), sum/avg the entries and make the
                                #  player 'pos' the one that is most played.
                                #  Could involve choosing which is the
                                #  destination entry so that the pos-feature
                                #  OneHotEncoding isn't effected.

    print("**** Data Modification: removeDuplicates - COMPLETE\t {} ({:.2}%) "
          "values effected.".format(count, count/(len(df))))

    return df


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

    # Count of entries effected by this process
    count = 0

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

                count = count + len(indicies[1:])


            # If there is only one player data entry, do not do anything.
            else:
                continue

    print("**** Data Modification: removeDuplicates - COMPLETE\t {} ({:.2}%) "
          "values effected.".format(count, count/(len(df)*len(df.columns))))

    return df


def modifyNanValues(df: pd.DataFrame,
                    NAN_LIMIT,
                    YEARS_PAIRS: list) -> pd.DataFrame:
    '''
    Replace all NaN values contained in the dataset with numeric values. The
    dataset's NaN values are handled for each year range so all adjustments.
    Values in the inputted dataset are replaced based on the indices sliced
    by the inputted list of year_pairs.
    would not affect a different year range.
    :param df: full ALL years dataset.
    :param NAN_LIMIT: The maximum amount of NaN values allowed until the
                        entire feature is made the same value.
    :return: full ALL years dataset.
    '''

    count = 0
    KEY_FEATURES = ['3P', '3PA', '3P%', 'FT%']

    # loop through each feature by year range. Based on how many Nan values
    # there are, change the Nan values.
    for YEARS in YEARS_PAIRS:
        df_year = df.loc[(df["Year"] >= YEARS[0]) & (df["Year"] <= YEARS[1])]

        for col in df.columns:
            nanCount = df_year[col].isnull().sum()

            '''
            Three situations:
            1) A column has significant Nan values, but its not in a commonly 
            Nan feature where Nan is shorthand for zero. Then replace all values 
            of that feature with -1 for that specific year range.
            2) A column has significant Nan values for features where Nan is 
            shorthand for zero. Then replace all Nan values with zero.
            3) A column has non-zero amount of Nan values but less than the 
            NanLimit. Then replace the Nan values with the median value of 
            the feature.
            '''
            if nanCount / len(df_year[col]) > NAN_LIMIT and \
                    col not in KEY_FEATURES:
                print("***** Replace column {} with a value of -1 for "
                      "years {}-{}.".format(col, YEARS[0], YEARS[1]))

                df.loc[df_year.index, col] = -1
                count = count + nanCount

            elif nanCount / len(df_year[col]) > NAN_LIMIT and \
                    col in KEY_FEATURES:
                df.loc[df_year.index, col] = \
                    df.loc[df_year.index, col].replace(np.nan,
                                                       df_year[col].median())
                count = count + nanCount

            elif nanCount != 0:
                df.loc[df_year.index, col] = \
                    df.loc[df_year.index, col].replace(np.nan,
                                                       df_year[col].median())
                count = count + nanCount

            else:
                continue

    print("**** Data Modification: Remove NaN values - COMPLETE\t {} ({:.2}%) "
          "values effected.".format(count, count/(len(df)*len(df.columns))))

    return df


def cleanPositionFeature(df: pd.DataFrame, THREE_POSITIONS_FLAG) -> \
        pd.DataFrame:
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

    count = 0

    for id in df['ID']:
        pos = df.loc[id, 'Pos']

        # 1) In each entry, only use the first position listed if a player is
        # listed with multiple positions. Ex: PG-SG, F-C, G-F, etc
        pos = df.loc[id, 'Pos']
        dash_position = pos.find('-')

        # If found dash char '-' is not found (-1) then do nothing
        # Cases
        #   Dash Position == 1 when position is listed as F-C
        #   Dash Position == 2 when position is listed as SG-SF
        if dash_position == 1:
            pos = pos[:1]
            count = count + 1
        elif dash_position == 2:
            pos = pos[:2]
            count = count + 1

        # 2) Only allow 3 or 5 cardinality for the 'position' feature
        if THREE_POSITIONS_FLAG:
            if pos == 'PG' or pos == 'SG':
                pos = 'G'
                count = count + 1
            elif pos == 'SF' or pos == 'PF':
                pos = 'F'
                count = count + 1
        else:
            if pos == 'G':
                pos = 'SG'
                count = count + 1
            elif pos == 'F':
                pos = 'SF'
                count = count + 1

        # Set the resulting position categorical.
        df.loc[id, 'Pos'] = pos


    print("**** Data Modification: Clean Player Position - COMPLETE\t {}"
          " ({:.2}%) values effected.".format(count,
                                              count/(len(df)*len(df.columns))))

    return df


def modifyData(df: pd.DataFrame, YEARS_PAIRS: list,
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

    YEARS = [YEARS_PAIRS[0][0], YEARS_PAIRS[len(YEARS_PAIRS)-1][1]]
    print("** Data Modification {}-{}: START".format(YEARS[0], YEARS[1]))

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
    # Ensure 'Position' feature has only 3 or 5 categories if included.
    df = cleanPositionFeature(df, THREE_POSITIONS_FLAG)

    ##########################
    # Consolidate any entries that are listed more than once.
    df = removeDuplicates2(df)

    ##########################
    # Remove nan features if over a criteria
    df = modifyNanValues(df, 0.3, YEARS_PAIRS)

    # Add in One-Hot Encoding 'Pos' feature values.
    df_oneHot_pos = pd.get_dummies(df['Pos'], prefix='Pos')
    df_oneHot_pos['ID'] = df['ID']

    df = pd.merge(df, df_oneHot_pos, how='left', on='ID')

    ##########################
    # Specific Player Filters
    count = 0

    # 2) Game filter
    count = df[(df['G'] >= REQ_GAMES)]
    df = df[(df['G'] >= REQ_GAMES)]
    print("**** Data Modification: GAME Filter - COMPLETE\t {}"
          " ({:.2}%) values effected.".format(count, count / len(df)))

    # 3) Time filter
    count = df[(df['MP'] >= REQ_GAMES * REQ_MIN)]
    df = df[(df['MP'] >= REQ_GAMES * REQ_MIN)]
    print("**** Data Modification: MINUTES Filter - COMPLETE\t {} ({:.2}%) "
          "values effected.".format(count, count / len(df)))

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

    df_model = modifyData(df_data, YEARS_PAIRS, REQ_GAMES, REQ_MIN,
                          THREE_POSITION_FLAG)

    df_model.to_csv("../data/ref/Season_Stats_MODEL_{}-{}.csv".format(
                    YEARS_PAIRS[0][0], YEARS_PAIRS[len(YEARS_PAIRS) - 1][1]),
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
