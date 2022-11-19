***
Directory: /data

Description:
Location where all input, output and reference dataset files should be placed or
 will be created after the 'main.py' script runs the functions
 loacted in 'DataPreparation.py' file. These include Data Quality Reports (DQR)
 and dataset .csv files for audit purposes.


How To:
There are two ways to run 'main.py':
    1) Create model data files (Requires INPUT files)
    2) Load a compatible csv file (a compatible file is created after running
    option one).

    *** It is recommended that a user runs option 1) first and then switches to
    option 2) to reduce computation time.


Procedure:
1) Create data files:
    The following csv files must be located in the /data/input/ directory.
        1. Season_Stats_<year1>_<year2>.csv - Player statistics from <year1> to
                            <year2> via BasketBall-Reference.com
        2. Players.csv - Player biographical information from 1950 to 2017
                            via Kaggle

    Run 'main.py' with the LOAD_MODEL_DATA flag set to False.

2) Load Existing data file:
    The following csv file must be located in the /data/ref/ directory.
        1. Season_Stats_MODEL_<year1>_<year2>.csv - Output of 'DataPreparation
                            .py' functions. Pre-processed data that is ready to
                            be used in various models in 'main.py'

    Run 'main.py' with the LOAD_MODEL_DATA flag set to True.

