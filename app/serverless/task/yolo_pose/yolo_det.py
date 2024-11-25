import io
import base64
import json

import cv2
import numpy as np
from yolo_det import YOLOposeWarp, warp_image

# Initialize your model
def init_context(context):
	context.logger.info('Init context...  0%')
	model = YOLOposeWarp()
	context.user_data.model_handler = model
	context.logger.info('Init context...100%')

# Inference endpoint
def handler(context, event):
	context.logger.info('Run custom yolov8 model')
	data = event.body
	threshold = data.get('threshold', 0.6)
	image_buffer = io.BytesIO(base64.b64decode(data['image']))
	image = cv2.imdecode(np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

	kpts, cls = context.user_data.model_handler.detector(image, conf_thres=threshold)

	return context.Response(body=json.dumps({
		'keypoints': kpts.tolist(),
		'class' : cls.tolist()
	}), headers={},
		content_type='application/json', status_code=200)