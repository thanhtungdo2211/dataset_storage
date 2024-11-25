import os
import uuid
import json
from typing import List

import numpy as np
from PIL import Image
from prefect import flow, task #type: ignore
from qdrant_client import QdrantClient #type: ignore
from qdrant_client.models import PointStruct, VectorParams, Distance #type: ignore

from configure import Config
import database.models as models
from etl.image_quality import Brightness, Blurriness, Entropy
from serverless.task.image_embeded_clip.image_feature import ImageFeatureExtractor
from storage.minio_storage import MinioClientWrapper
from utils.utils import get_file_download_date, crop_image
from ultralytics import YOLO

class ImageProcessor:
    def __init__(self, minio_config, qdrant_url):
        self.auto_label = YOLO('/home/mq/data_disk2T/Data-Recall-System/weights/best.pt')
        self.minio_client = MinioClientWrapper(minio_config['domain'], minio_config['user'], minio_config['password'])
        self.client = QdrantClient(url=qdrant_url, timeout=60.0)
        self.feature_extractor = ImageFeatureExtractor()
        self.feature_size = 512
        self._initialize_qdrant_collections()
 
    def _initialize_qdrant_collections(self):
        try:
            collections = {
                "object_collection": "Object features collection",
                "image_collection": "Image features collection",
                "object_description_collection": "Object description collection"
            }
            for collection_name in collections:
                if not self.client.collection_exists(collection_name=collection_name):
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=self.feature_size, distance=Distance.COSINE)
                    )
        except Exception as e:
            print(f"An error occurred while initializing Qdrant collections: {e}")

    def close(self):
        pass 

@task(name="Extract features")
def extract_features(img, ip: ImageProcessor):
    return ip.feature_extractor.extract_features_clip(img).tolist()

@task(name="Save image to Qdrant", retries=3)
def save_image_to_qdrant(collection_name: str, points: List[PointStruct], ip: ImageProcessor):
    try:
        ip.client.upsert(collection_name=collection_name, points=points)
    except Exception as e:
        print(f"An error occurred while uploading to Qdrant: {e}")

@task(name="Upload to Minio", retries=3)
def upload_to_minio(file_path, file_name, task_name, minio_config):
    minio_client = MinioClientWrapper(minio_config['domain'], minio_config['user'], minio_config['password'])
    minio_client.upload_object(bucket_name=task_name, file_path=file_path, object_name=file_name)
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": "*",
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{task_name}/*"]
            }
        ]
    }
    policy_str = json.dumps(policy)
    minio_client.set_bucket_policy(task_name, policy_str)
    url_image = minio_client.get_url_object(bucket_name=task_name, object_name=file_name)
    url_image_shorted = ""
    for item in url_image.split("?")[0].split('://')[1].split('/')[1:]:
            url_image_shorted += '/' + item
    return url_image_shorted

@task(name="Get metrics", retries=3)
def calculate_metrics(img):
    dark_score = Brightness.calculate_brightness_score(img)['brightness_perc_95']
    blur_score = Blurriness.calculate_blurriness_score(img)
    light_score = 1 - Brightness.calculate_brightness_score(img)['brightness_perc_5']
    low_information_score = Entropy.calc_entropy_score(img)
    metrics = {
        "dark_score": dark_score,
        "light_score": light_score,
        "low_information_score": low_information_score,
        "blur_score": blur_score
    }
    return metrics

@task(name="Process image", retries=3)
def process_image(file_path, file_name, task_name, minio_config, qdrant_url):
    processor = ImageProcessor(minio_config, qdrant_url)
    img = Image.open(file_path)
    width, height = img.size

    url_image = upload_to_minio(file_path, file_name, task_name, minio_config)

    id_image = uuid.uuid4().hex
    feature_vector = extract_features(img, processor)

    qdrant_image = PointStruct(
        id=id_image,
        vector=feature_vector,
        payload={"id_image": id_image}
    )
    
    save_image_to_qdrant("image_collection", [qdrant_image], processor)
    description = "Description: "
    date_time = get_file_download_date(file_path)
    metrics = calculate_metrics(img)
    metadata = {
        "date_time": date_time,
        "local_path": file_path,
        "task": task_name,
        "size": "{}x{}".format(width, height)
    }

    objects = processor.auto_label(img)
    
    id_objects = []
    features = []
    for bbox in objects[0].boxes.xyxy:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
   
        cr_img = crop_image(img, (x1, y1, x2, y2))
        cr_ft = np.array(extract_features(cr_img, processor)).tolist()
        features.append(cr_ft)
        
        id_object = uuid.uuid4().hex
        id_objects.append(id_object)
        
    qdrant_objects = [
        PointStruct(
            id=id_object,
            vector=feature,
            payload={"id_image": id_image, "id_object": id_object}
        )
        for feature, id_object in zip(features, id_objects)
    ]
    save_image_to_qdrant(collection_name="object_collection", points=qdrant_objects, ip=processor)
    processor.close()

@task(name="Create session database")
def create_config():
    minio_config = {
        'domain': Config.minio.MINIO_DOMAIN,
        'user': Config.minio.MINIO_USER,
        'password': Config.minio.MINIO_PASSWORD
    }
    qdrant_url = f"http://{Config.qdrant.QDRANT_HOST}:{Config.qdrant.QDRANT_PORT}"
    return minio_config, qdrant_url

@task(name="Process images in folder")
def process_images_in_folder(folder_path, task_name, minio_config, qdrant_url):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            process_image(file_path, file_name, task_name, minio_config, qdrant_url)

@flow(name="Main Process")
def main_process():
    minio_config, qdrant_url = create_config()
    process_images_in_folder("/home/mq/data_disk2T/Data-Recall-System/images/test", "test", minio_config, qdrant_url)

if __name__ == "__main__":
    main_process()
