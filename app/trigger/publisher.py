import os
import redis, json
from etl.label_quality import CleanLabObjectDetection

def process_and_publish(label_folder_path, predict_folder_path, image_folder_path, num_classes, threshold):
    processed_files = set()

    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    label_quality = CleanLabObjectDetection()
    img_paths, label_paths, predict_paths = label_quality.clean_lap(label_folder_path, predict_folder_path, image_folder_path, num_classes, threshold)
    
    for img_path, label_path, predict_path in zip(img_paths, label_paths, predict_paths):
        if img_path in processed_files:
            continue
        message = json.dumps({
            "image_path": img_path,
            "label_path": label_path,
            "predict_path": predict_path,
        })
        r.publish("file_differences", message)
        processed_files.add(img_path)

# Example usage
# if __name__ == "__main__":
#     process_and_publish(label_folder_path, predict_folder_path, image_folder_path, num_classes, threshold)