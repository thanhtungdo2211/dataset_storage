import sys
sys.path.append('serverless/task/image_embeded_clip/')
from onnx_clip import OnnxClip
import os
import numpy as np
from PIL import Image
    
class ImageFeatureExtractor:
    def __init__(self):
        #self.onnx_model = OnnxClip(batch_size=16, cache_dir="onnx_clip/data")
        self.onnx_model = OnnxClip(batch_size=16, cache_dir=".cache")
        
    def extract_features_clip(self, image: Image):
        """Extract features from an image using CLIP model."""
        image = np.array(image)
        image = [image]
        image_features = self.onnx_model.get_image_embeddings(image)
        norm = np.linalg.norm(image_features, axis=-1, keepdims=True)
        image_features /= norm

        flattened_features = image_features.flatten()
        normalized_features = flattened_features / np.linalg.norm(flattened_features)
        
        return normalized_features
    
    def features_to_string(self, features_array):
        features_string = np.array2string(features_array, separator=',', max_line_width=np.inf, floatmode='maxprec').strip('[]')
        return features_string


