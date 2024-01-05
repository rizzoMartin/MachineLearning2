import pandas as pd
from pandas import DataFrame
from pymongo import MongoClient
from sqlalchemy import create_engine

# GET THE DATA FROM MYSQL WITH THE STORED PROCEDURE

engine = create_engine('mysql+mysqlconnector://root:root@localhost/hotel_reviews', pool_recycle=3600, pool_size=5)
procedure = 'select_all'

print('getting data from db...')
raw_conn = engine.raw_connection()
cur = raw_conn.cursor()
cur.callproc(procedure)
for result in cur.stored_results():
    df = DataFrame(result.fetchall())
column_names_list = [i[0] for i in result.description]
raw_conn.close()

df.columns = column_names_list

print(df.head())

# SAVE THE DATA IN MONGO

client = MongoClient('localhost:27017')
db = client.hotels
db.reviews.insert_many(df.to_dict('records'))