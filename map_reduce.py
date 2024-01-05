from pymongo import MongoClient
from bson.code import Code

# MAP REDUCE based on hotel name count the number of reviews

db = MongoClient('localhost:27017').hotels

map = Code("""
        function() {
            emit(this.hotel, 1)
        }
        """)

reduce = Code("""
            function(key, values) {
                var total = 0;
                for (var i = 0; i < values.length; i++) {
                    total += values[i]
                }
                return total;
            }
""")

result = db.reviews.map_reduce(map, reduce, 'results')

for doc in result.find():
    print(doc)