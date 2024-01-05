import pandas as pd
from pandas import DataFrame
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

from nltk.corpus import stopwords
stop = stopwords.words('english')

df['review_without_stopwords'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

print(df.head())

df.to_sql(name='reviews2',con=engine,if_exists='fail',index=False,chunksize=1000)