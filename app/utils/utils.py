import os, datetime
import numpy as np

def get_file_download_date(file_path):
    file_stats = os.stat(file_path)
    access_time = file_stats.st_atime
    download_date = datetime.datetime.fromtimestamp(access_time)
    return str(download_date)

def crop_image(image, crop_box):
    cropped_image = image.crop(crop_box)
    return cropped_image

def convertToYolo(labels, image_width, image_height):
    yolo_labels = []
    for item in labels:
        for key in item.keys():
            x1, y1, x2, y2 = item[key]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1

            center_x /= image_width
            center_y /= image_height
            width /= image_width
            height /= image_height

            yolo_label = f"{key} {center_x} {center_y} {width} {height}"
            yolo_labels.append(yolo_label)
    return yolo_labels
