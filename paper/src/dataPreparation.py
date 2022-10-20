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

##############################
# Load dataset

DATA_PATH = "../data/Seasons_Stats.csv"

df_season = pd.read_csv(DATA_PATH)

##############################
# Clean data based on visual observation

# 1) Remove 'blank' columns  {blanl, blank2}
df_season = df_season.drop(columns={'blanl', 'blank2'})

# 2) Remove 'id' column  {'Unnamed: 0'}
df_season = df_season.drop(columns={'Unnamed: 0'})

##############################
# Output a Data Quality Report for analysis

from vt_ApplicationsOfML.Libraries.DataExploration.DataQualityReport import DataQualityReport

OUTPUT_PATH = "../data/dqr_ALL_Season_Stats.csv"
NON_NUMERIC_COLUMNS = ['Player', 'Tm', 'Pos', 'Unnamed: 0']

report_all = DataQualityReport()
report_all.quickDQR(df_season, df_season.columns, NON_NUMERIC_COLUMNS)

if OUTPUT_FILES:
    report_all.to_csv(OUTPUT_PATH)

##############################
    #####################
##############################

##############################
# Modify data based on specified conditions
#   1) Player must have played in the following years: 2000 - 2009.
#   2) Player must have played in at least 20 games.
#   3) Player must have played at least 10 minutes per game played.
##############################

# 1) Year filter
df_season = df_season[(df_season['Year'] >= 2000) & (df_season['Year'] < 2010)]

# 2) Game filter
df_season = df_season[(df_season['G'] >= 20)]

# 3) Time filter
df_season = df_season[(df_season['MP'] >= 20*10)]


##############################
# Output a Data Quality Report for filtered data

OUTPUT_PATH = "../data/dqr_FILTERED_Season_Stats.csv"
NON_NUMERIC_COLUMNS = ['Player', 'Tm', 'Pos', 'Unnamed: 0']

report = DataQualityReport()
report.quickDQR(df_season, df_season.columns, NON_NUMERIC_COLUMNS)

if OUTPUT_FILES:
    report.to_csv(OUTPUT_PATH)

'''
NOTES - Filtered players data
1) Almost all of the features have filled in data. 'n_missing' and 'n_zero' 
are pretty low for nearly all features. Unlike the data for ALL years.
2) 'Pos' has a cardinality of 16. That has to change.
3) ALL data has 24691, and 2000-2009 has 4242 players.
4) Some data features like BPM and BLK started in 1973-74
5) Some features such as OBPM and DBPM can be negative. This is ok.
'''