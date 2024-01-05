from pymongo import MongoClient
import pandas as pd

db = MongoClient('localhost:27017').hotels
reviews = db.reviews
df = pd.DataFrame(list(reviews.find()))

df.to_csv('../csvs/data.csv')