from typing import List, Optional
import sys
sys.path.append('/home/mq/data_disk2T/Data-Recall-System/app/serverless/task/autolabel')

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from florence import AutoLabel_FLorence2
from sam import GroundingDino_RAM

class ObjectLabeler:
    def __init__(self, florence_model_path, grounding_dino_config, ram_plus_checkpoint, grounded_checkpoint) -> None:
        self.flo = AutoLabel_FLorence2(florence_model_path)
        self.dino = GroundingDino_RAM(grounding_dino_config, ram_plus_checkpoint, grounded_checkpoint)

    @staticmethod
    def calculate_iou(boxa: str, boxb: str) -> float:
        x1_min, y1_min, x1_max, y1_max = boxa
        x2_min, y2_min, x2_max, y2_max = boxb
        
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        inter_width = max(0, x_inter_max - x_inter_min + 1)
        inter_height = max(0, y_inter_max - y_inter_min + 1)
        intersection_area = inter_width * inter_height
        
        boxa_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
        boxb_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)
        
        union_area = boxa_area + boxb_area - intersection_area
        
        iou = intersection_area / union_area
        return iou

    def get_description(self, image: Image):
        return self.flo.get_description(image)
    
    def get_less_description(self, image: Image):
        return self.flo.get_less_description(image)
    
    def label_all_objects(self, image: Image) -> List[dict]:
        labels = []
        labels.extend(self.flo.label_all_objects(image))
        labels.extend(self.dino.label_all_objects(image))        
        res = []
        a = set()
        for item in labels:
            arr = []
            for i in item.keys():
                arr.append({i : item[i]})
                a.add(i)
        for item in a:
            arr = []
            for i in labels:
                if item in i.keys():
                    arr.append(i[item])
            res.append(arr)
        keys = []
        for item in a:
            keys.append(item)
        final_res = []
        for idx, item in enumerate(res):
            key = keys[idx]
            for i in range(len(item) - 1):
                delete = []
                for j in range(i + 1, len(item)):
                    boxa = item[i] 
                    boxb = item[j]
                    if self.calculate_iou(boxa, boxb) > 0.7:
                        delete.append(j)
                for k in sorted(delete, reverse=True):
                    del item[k]
            for t in item:
                final_res.append({key: t})
        return final_res

    def label_by_keywords(self, image : Image, key: List[str]) -> Optional[List[str]]:
        result = []
        keyword = ", ".join(key)
        result.extend(self.flo.auto_label(image, key))

        arr = self.dino.grounding_dino_inference(image, keyword)
        result.extend(arr)
        labels = result
        res = []
        keys = set()

        for item in labels:
            keys.update(item.keys())

        for item in keys:
            group = [i for i in labels if item in i]
            res.append(group)

        final_res = []
        for group in res:
            i = 0
            while i < len(group):
                j = i + 1
                while j < len(group):
                    boxa = list(group[i].values())[0]  
                    boxb = list(group[j].values())[0]  
                    if AutoLabel_FLorence2.calculate_iou(boxa, boxb) > 0.7:
                        group.pop(j)  
                    else:
                        j += 1
                final_res.append(group[i])
                i += 1

        return final_res

