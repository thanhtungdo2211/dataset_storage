import sys
sys.path.append('serverless/task/text_embeded_clip/')

from onnx_clip import OnnxClip
import os
import numpy as np

class TextFeatureExtractor:
    def __init__(self):
        #self.onnx_model = OnnxClip(batch_size=16, cache_dir="onnx_clip/data")
        self.onnx_model = OnnxClip(batch_size=16, cache_dir=".cache")
        
    def extract_features_from_text(self, text: str):
        """Extract features from text using CLIP model."""
        text = [text]
        text_features = self.onnx_model.get_text_embeddings(text)
        
        norm = np.linalg.norm(text_features, axis=-1, keepdims=True)
        text_features /= norm

        flattened_features = text_features.flatten()
        normalized_features = flattened_features / np.linalg.norm(flattened_features)
        
        return normalized_features

    def features_to_string(self, features_array):
        features_string = np.array2string(features_array, separator=',', max_line_width=np.inf, floatmode='maxprec').strip('[]')
        return features_string