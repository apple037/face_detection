from pymongo import MongoClient
import config
import numpy as np


def list_to_ndarray(data_list):
    return np.array(data_list)


class MongoDb:
    def __init__(self, collection_name):
        self.client = MongoClient(config.MONGODB_CONNECTION)
        self.db = self.client[config.MONGODB_DB]
        self.collection = self.db[collection_name]

    def insert(self, data):
        self.collection.insert_one(data)

    def find(self, data):
        return self.collection.find_one(data)

    def find_all(self):
        return self.collection.find()

    def update(self, data, new_data):
        self.collection.update_one(data, new_data)

    def delete(self, data):
        self.collection.delete_one(data)

    def delete_all(self):
        self.collection.delete_many({})

    def close(self):
        self.client.close()
