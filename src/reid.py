# src/reid.py

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity

class ReIdentifier:
    def __init__(self):
        self.feature_db = {}  # custom_id: embedding vector
        self.next_id = 1

        # Load pretrained ResNet50 model for feature extraction
        resnet = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC layer
        self.model.eval()

        # Transformation: resize, normalize
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return np.zeros(2048)

        try:
            input_tensor = self.preprocess(cropped).unsqueeze(0)  # Add batch dim
            with torch.no_grad():
                features = self.model(input_tensor).squeeze().numpy()
            return features
        except Exception:
            return np.zeros(2048)

    def update(self, tracks, frame):
        updated_tracks = []

        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            feat = self.extract_features(frame, [x1, y1, x2, y2])

            matched_id = None
            for known_id, known_feat in self.feature_db.items():
                similarity = cosine_similarity([feat], [known_feat])[0][0]
                if similarity > 0.85:  # Higher threshold for deep features
                    matched_id = known_id
                    break

            if matched_id is not None:
                track['id'] = matched_id
            else:
                track['id'] = self.next_id
                self.feature_db[self.next_id] = feat
                self.next_id += 1

            updated_tracks.append(track)

        return updated_tracks
