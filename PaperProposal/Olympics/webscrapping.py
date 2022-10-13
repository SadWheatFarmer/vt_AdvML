'''
File:
Author:
Date:
Desc:   Collect athletic performance information about the history of the
        Decathlon event of the Olympics and Trank & Field championships.

        Data from WorldAthletics.org
'''

'''
Status
- Can gather all hyperlinks to access each Olympic games
- Can go through a specific Olympic game and pull headers
- Can go through a specific Olympic game and include all non-class data.
- CANNOT join a dataframe of a specific Olympic game and a bigger dataframe.
--  Consider a map of dataframes for now
--  Consider removing certain columns that could be redundent (AKA medal, 
we already have a column for what RANK the athlete placed in.

TODOs

3) Create a 'for' loop to process ALL pages
p1) Functionalize the code in 'webscrapping.py'
p2) Repeat steps for the women's heptathlon 
p3) Repeat for given data about athletes
'''


# Import packages (beautifulsoup and Requests) to interact with HTTP pages.
from bs4 import BeautifulSoup
import requests

# Import package helpful for data interaction
import pandas as pd

###
# Define file level definitions
webpages = [
            'https://www.worldathletics.org/records/all-time-toplists'
            '/combined-events/decathlon/outdoor/men/senior?regionType=world'
            '&windReading=all&page=1&bestResultsOnly=true&firstDay=1899-12-30&lastDay=2021-08-17',
            'https://www.worldathletics.org/records/all-time-toplists'
            '/combined-events/decathlon/outdoor/men/senior?regionType=world'
            '&windReading=all&page=2&bestResultsOnly=true&firstDay=1899-12-30&lastDay=2021-08-17',
            'https://www.worldathletics.org/records/all-time-toplists'
            '/combined-events/decathlon/outdoor/men/senior?regionType=world'
            '&windReading=all&page=3&bestResultsOnly=true&firstDay=1899-12-30&lastDay=2021-08-17',
            'https://www.worldathletics.org/records/all-time-toplists'
            '/combined-events/decathlon/outdoor/men/senior?regionType=world'
            '&windReading=all&page=4&bestResultsOnly=true&firstDay=1899-12-30&lastDay=2021-08-17'
            ]

webpages_olympedia = [
    'https://www.olympedia.org/event_names/93'
]

###

# Get the collection of data tables from Olympedia
# Collect top-level table information from the website (year, data url)
page = requests.get(webpages_olympedia[0])
html_src = page.content

# Extract the links from the HTML Table. Remove all other HTML information
soup = BeautifulSoup(html_src, 'html.parser')
table_OlypicsOverall = soup.find("table", {"class": "biodata"})
table_games = soup.find("table", {"class": "table table-striped"})

results_urls = []
for a in table_games.find_all("a", href=True):
    href = a.get('href')
    if "results" in href:
        results_urls.append(href)
        print("HREF = {}".format(href))

# Go to each Olympic results URL and extract participant information
URL_BASE = "https://www.olympedia.org/"

# For each olympic games linked, extract all recorded athlete performance data
df_athletes = pd.DataFrame

setHeader = True
for ext in results_urls:
    url = URL_BASE + ext

    page = requests.get(url)
    html_src = page.content
    soup = BeautifulSoup(html_src, 'html.parser')

    # 1) Get the year of the Olympics
    table_overall = soup.find("table", {"class": "biodata"})

    # first td is the Date (but check to make sure)
    if 'Date' in table_overall.find('tr').text:
        date = table_overall.find('td').text
        olympic_year = date[len(date)-4:len(date)]

    # 2) Get the performance of athletes in that Olympic games
    table_participants = soup.find("table", {"class": "table table-striped"})

    # Set the dataframe header for the specific Olympic games
    header = []
    points_flag = True
    used_cols = []
    counter = 0
    for col in table_participants.find_all('th'):
        # Only using the 1st 'point' column
        # Don't include the blank columns for MEDAL, WR/OR, or references
        if "Points" in col.text and points_flag:
            header.append(col.text)
            points_flag = False
            used_cols.append(counter)
            counter += counter + 1
        elif "Points" not in col.text:
            header.append(col.text)
            used_cols.append(counter)
            counter += counter + 1


        if "1,500 metres" in col.text:
            break

    # Extract all data from the Olympic games
    tbody = table_participants.find_all('tr')
    for row in tbody[1:len(tbody)]:
                # TODO - How do you make sure that the headers are for the
                #  right columns no matter what Olympic competition?
                #   You make dataframes for each Olympics then JOIN them

        item_list = []
        print(row.find_all('td'))
        for item in row.find_all('td'):
            if len(item.select('a[class]')) == 0:
                item_list.append(item.text)
            else:
                header.pop()    # TODO - Assumes that only the LAST columns
                                # on a page will have the class object.

        df_game = pd.DataFrame([item_list], columns=header)
        #df_athletes = pd.concat([df_athletes, df_game], join="outer")
        df_athletes.join(other=df_game)







########
# Webscrapping for WorldAthletic.org (Non-Olympic Decathlon dataset)
# # TODO - 3)
# # Collect table information from the website
# page = requests.get(webpages[0])
# html_src = page.content
#
# # Extract just the HTML Table. Remove all other HTML information
# soup = BeautifulSoup(html_src, 'html.parser')
# records = soup.find("table", {"class": "records-table"})
#
# #print("Records Table = \n{}".format(records))
#
# # Parse the table for HEADER information
#
# th_list = records.find_all("th")
# #print("Records Table HEADER = \n{}".format(th_list))
# #print("Records Table HEADER size: \n{}".format(th_list))
#
# header = []
# for x in range(len(th_list)):
#     text = th_list[x].text.replace(" ", "")
#     text = text.replace("\n", "")
#
#     header.append(text)
#
# header.append(" ")
# header.append("EventScore")
#
# #TODO - 1)
#
# df_data = pd.DataFrame(columns=header)
# print(df_data.head())
#
#
#
# # Parse the original html table for ATHLETE and PERFORMANCE information
#
# td_list = records.find_all("td")
#
# ITEMS_PER_ATHLETE = len(header)
#
# athlete = [None]*ITEMS_PER_ATHLETE
# #df_data.loc[len(df_data.index)] = athlete
# #print(df_data.head())
#
#
# for y in range(len(td_list)):
#     if y%ITEMS_PER_ATHLETE == 0:
#         df_data.loc[len(df_data.index)] = athlete
#         print(df_data.head())
#
#     entry = td_list[y].text.replace(" ", "")
#     entry = entry.replace("\n", "")
#     df_data.iat[int(y/ITEMS_PER_ATHLETE), y%ITEMS_PER_ATHLETE] = entry
#
#     #if y < ITEMS_PER_ATHLETE:
#         #print(entry)
#
# #TODO - 2)
# df_data.to_csv("decathlon_data.csv")






