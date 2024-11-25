import io
import base64
import json

import cv2
import numpy as np

from app.serverless.task.autolabel.test import AutoLabel_FLorence2

def init_context(context):
    context.logger.info("Init VLM model")
    model = AutoLabel_FLorence2()
    context.user_data.model_handler = model
    context.logger.info('Init model VLM...100%')

def handler(context, event):
    context.logger.info("Run  model")
    data = event.body
    text = data.get('text', "")
    image_buffer = io.BytesIO(base64.b64decode(data['image']))
    image = cv2.imdecode(np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    # Extract features
    label = context.user_data.model_handler.label_image(image, text = text)
    
    return context.Response(body=json.dumps({label}), headers={},
                            content_type='application/json', status_code=200)