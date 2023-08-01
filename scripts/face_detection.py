import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn import CosineSimilarity
from torchvision import transforms


class FaceRecognition:
    def __init__(self):
        self.device_one = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.aligner = MTCNN(min_face_size=90)  # min_face_size=80
        self.facenet_preprocess = transforms.Compose([transforms.Resize((160, 160))])
        # skip whitening
        # self.facenet_preprocess = transforms.Compose([modules.Whitening()])
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

        print('[INFO] FaceRecognition initialized')

    def get_face_embedding(self, img):
        img = self.aligner(img)
        if img is None:
            return None
        img = self.facenet_preprocess(img)
        img = img.unsqueeze(0).to(self.device_one)
        return self.facenet(img).detach().cpu().numpy()[0]

    def get_cosine_similarity(self, embedding1, embedding2):
        embedding1 = torch.from_numpy(embedding1)
        embedding2 = torch.from_numpy(embedding2)
        if embedding1 is None or embedding2 is None:
            print("Empty embedding.")
            return None
        cos = CosineSimilarity(dim=0, eps=1e-6)
        score = cos(embedding1, embedding2)
        return np.array(score)
