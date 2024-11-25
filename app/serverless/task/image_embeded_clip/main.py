import io
import base64
import json

import cv2
import numpy as np

from image_feature import ImageFeatureExtractor

def init_context(context):
    context.logger.info("Init CLIP model")
    model = ImageFeatureExtractor()
    context.user_data.model_handler = model
    context.logger.info('Init context...100%')

def handler(context, event):
    context.logger.info("Run CLIP model")
    data = event.body
    image_buffer = io.BytesIO(base64.b64decode(data['image']))
    image = cv2.imdecode(np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    # Extract features
    features = context.user_data.model_handler.extract_features_clip(image)
    
    # Example: Convert features to serializable format
    features_serializable = context.user_data.model_handler.features_to_string(features)
    
    return context.Response(body=json.dumps({"features": features_serializable}), headers={},
                            content_type='application/json', status_code=200)