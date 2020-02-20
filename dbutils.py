import psycopg2
import sys, os
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from datetime import datetime, timedelta
import json

unix_ts = 1507126064
PGHOST = "localhost"
PGDATABASE = "datamining"
PGUSER = "postgres"
PGPASSWORD = "root"

CREATE_TABLE_QUERY = "CREATE TABLE IF NOT EXISTS matches ( match_id bigint , start_time TIMESTAMP, ranked BOOLEAN, captain_mode BOOLEAN, p1_hero INTEGER,p2_hero INTEGER,p3_hero INTEGER, p4_hero INTEGER,p5_hero INTEGER,p6_hero INTEGER,p7_hero INTEGER,p8_hero INTEGER,p9_hero INTEGER,p10_hero INTEGER, win BOOLEAN,raw_match_details VARCHAR, PRIMARY KEY (match_id)); "
CONNECTION_STRING = "host=" + PGHOST + " port=" + "5432" + " dbname=" + PGDATABASE + " user=" + PGUSER + " password=" + PGPASSWORD


# match_id , start_time , ranked , captain_mode , p1_hero ,p2_hero ,p3_hero , p4_hero ,p5_hero ,p6_hero ,p7_hero ,p8_hero ,p9_hero ,p10_hero , win ,raw_match_details


def createTable(conn):
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_QUERY)
    # print("Table created successfully")



def insertMatchDetail(match_details):
    match_id = match_details["match_id"]
    start_time = (datetime.fromtimestamp(match_details["start_time"]) - timedelta(hours=2)).strftime(
        '%Y-%m-%d %H:%M:%S')
    ranked = match_details["lobby_type"] == 7  # "lobby_type": 7, "game_mode": 4
    captain_mode = match_details["game_mode"] == 2
    # "player_slot": 0, 1 2 3 4  128,129,130,131,132
    for player in match_details['players']:
        if player["player_slot"] == 0:
            p1_hero = player["hero_id"]
        elif player["player_slot"] == 1:
            p2_hero = player["hero_id"]
        elif player["player_slot"] == 2:
            p3_hero = player["hero_id"]
        elif player["player_slot"] == 3:
            p4_hero = player["hero_id"]
        elif player["player_slot"] == 4:
            p5_hero = player["hero_id"]
        elif player["player_slot"] == 128:
            p6_hero = player["hero_id"]
        elif player["player_slot"] == 129:
            p7_hero = player["hero_id"]
        elif player["player_slot"] == 130:
            p8_hero = player["hero_id"]
        elif player["player_slot"] == 131:
            p9_hero = player["hero_id"]
        elif player["player_slot"] == 132:
            p10_hero = player["hero_id"]
        else:
            print("ERROR parsing player slot!!!!! match_id", match_id)
    win = match_details["radiant_win"]
    raw_match_details = match_details
    insertMatch(match_id, start_time, ranked, captain_mode, p1_hero, p2_hero, p3_hero, p4_hero, p5_hero, p6_hero,
                p7_hero, p8_hero, p9_hero, p10_hero, win, raw_match_details)


def insertMatch(match_id, start_time, ranked, captain_mode, p1_hero, p2_hero, p3_hero, p4_hero, p5_hero, p6_hero,
                p7_hero, p8_hero, p9_hero, p10_hero, win, raw_match_details):
    conn = psycopg2.connect(CONNECTION_STRING)
    createTable(conn)# make sure table exits if not create it
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO matches (match_id , start_time , ranked , captain_mode , p1_hero ,p2_hero ,p3_hero , p4_hero ,p5_hero ,p6_hero ,p7_hero ,p8_hero ,p9_hero ,p10_hero , win ,raw_match_details) VALUES ({},'{}'::timestamp,{},{},{},{},{},{},{},{},{},{},{},{},{},'{}') ON CONFLICT (match_id) DO NOTHING;".format(
            match_id, start_time, ranked, captain_mode, p1_hero, p2_hero, p3_hero, p4_hero, p5_hero, p6_hero, p7_hero,
            p8_hero, p9_hero, p10_hero, win, json.dumps(raw_match_details)));
    # print("row inserted successfully")

    conn.commit()
    conn.close()


# Load the data
def load_data():
    sql_command = "SELECT  p1_hero, p2_hero, p3_hero, p4_hero, p5_hero, p6_hero, p7_hero, p8_hero, p9_hero, p10_hero, win FROM matches;"
    conn = psycopg2.connect(CONNECTION_STRING)
    data = pd.read_sql(sql_command, conn)
    print(data.shape)
    conn.close()
    return data

# result =
# for firstname, lastname in result.getresult() :
#     print firstname, lastname
# cur = con.cursor()
# cur.execute("SELECT admission, name, age, course, department from STUDENT")
# rows = cur.fetchall()
#
# for row in rows:
#     print("ADMISSION =", row[0])
#     print("NAME =", row[1])
#     print("AGE =", row[2])
#     print("COURSE =", row[3])
#     print("DEPARTMENT =", row[4], "\n")
