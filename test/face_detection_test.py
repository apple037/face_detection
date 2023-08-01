import unittest

from scripts import face_detection
from PIL import Image
from scripts import mongodb
from scripts.mongodb import list_to_ndarray


class MyTestCase(unittest.TestCase):
    def test_get_embedding(self):
        face = face_detection.FaceRecognition()
        img = Image.open('../test.png')
        embedding = face.get_face_embedding(img)
        if embedding is not None:
            print("Face embedding:")
            print(embedding)
        else:
            print("No face detected in the image.")

    def test_get_similarity(self):
        face = face_detection.FaceRecognition()
        img1 = Image.open('../test.png')
        img2 = Image.open('../test2.png')
        img3 = Image.open('../test3.png')
        embedding1 = face.get_face_embedding(img1)
        embedding2 = face.get_face_embedding(img2)
        embedding3 = face.get_face_embedding(img3)
        # 1 and 2 should get a bad score
        print("Similarity between 1 and 2:", face.get_cosine_similarity(embedding1, embedding2))
        # 1 and 3 should get a bad score
        print("Similarity between 1 and 3:", face.get_cosine_similarity(embedding1, embedding3))
        # 2 and 3 should get a good score
        print("Similarity between 2 and 3:", face.get_cosine_similarity(embedding2, embedding3))

    def test_embedding_in_mongo(self):
        face_collection = mongodb.MongoDb('faces')
        img1 = Image.open('../test2.png')
        embedding1 = face_detection.FaceRecognition().get_face_embedding(img1)
        face_collection.insert({'name': 'test2', 'embedding': embedding1.tolist()})
        # test read from mongo
        face = face_collection.find({'name': 'test2'})
        print(face['embedding'])

    def test_get_similarity_from_mongo(self):
        face_collection = mongodb.MongoDb('faces')
        face1 = face_collection.find({'name': 'test1'})
        array1 = list_to_ndarray(face1['embedding'])
        face2 = face_collection.find({'name': 'test2'})
        array2 = list_to_ndarray(face2['embedding'])
        print("Similarity between 1 and 2:",
              face_detection.FaceRecognition().get_cosine_similarity(array1, array2))
