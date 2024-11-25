from minio import Minio #type: ignore
from minio.error import S3Error #type: ignore
import os 

class MinioClientWrapper:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        print(f"{endpoint}:9000", access_key, secret_key)
        self.client = Minio(f"{endpoint}:9000", access_key=access_key, secret_key=secret_key, secure=secure)

    def list_buckets(self):
        buckets = self.client.list_buckets()
        return [bucket.name for bucket in buckets]

    def list_objects(self, bucket_name):
        objects = self.client.list_objects(bucket_name)
        for obj in objects:
            print(obj.object_name, obj.last_modified, obj.etag, obj.size, obj.content_type)

    def get_object(self, bucket_name, object_name):
        try:
            data = self.client.get_object(bucket_name, object_name)
            with open(object_name, "wb") as file_data:
                for d in data:
                    file_data.write(d)
        except S3Error as exc:
            print("Error occurred:", exc)

    def get_url_object(self, bucket_name, object_name):
        try:
            url = self.client.presigned_get_object(bucket_name, object_name)
            return url
        except S3Error as exc:
            print("Error occurred:", exc)

    def upload_object(self, bucket_name, object_name, file_path):
        try:
            found = self.client.bucket_exists(bucket_name)
            if not found:
                self.client.make_bucket(bucket_name)
            self.client.fput_object(bucket_name, object_name, file_path)
        except S3Error as exc:
            print("Error occurred:", exc)

    def remove_object(self, bucket_name, object_name):
        try:
            self.client.remove_object(bucket_name, object_name)
        except S3Error as exc:
            print("Error occurred:", exc)

    def set_bucket_policy(self, bucket_name, policy):
        try:
            self.client.set_bucket_policy(bucket_name, policy)
        except S3Error as exc:
            print("Error occurred:", exc)