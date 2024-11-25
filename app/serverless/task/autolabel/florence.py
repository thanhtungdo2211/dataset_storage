import sys
sys.path.append('/home/mq/data_disk2T/Data-Recall-System/app/serverless/task/autolabel')

from transformers import AutoModelForCausalLM, AutoProcessor
import supervision as sv
from PIL import Image

class AutoLabel_FLorence2:
    def __init__(self, model_name: str):
        self.checkpoint = model_name # microsoft/Florence-2-large or microsoft/Florence-2-base
        self.device = 'cpu'
        self.model, self.processor = self.load_model()

    def load_model(self):
        try:
            model = AutoModelForCausalLM.from_pretrained(self.checkpoint, trust_remote_code=True).to(self.device)
            processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
            return model, processor
        except Exception as e:
            print(f"Error loading model and processor: {e}")
            return None, None

    @staticmethod
    def calculate_iou(boxa, boxb):
        x1_min, y1_min, x1_max, y1_max = boxa
        x2_min, y2_min, x2_max, y2_max = boxb

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        inter_width = max(0, x_inter_max - x_inter_min + 1)
        inter_height = max(0, y_inter_max - y1_min + 1)
        intersection_area = inter_width * inter_height

        boxa_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
        boxb_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

        union_area = boxa_area + boxb_area - intersection_area

        iou = intersection_area / union_area
        return iou
    
    def run_inference(self, image: Image, task: str, text: str = ""):
        try:
            prompt = task + text
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            return self.processor.post_process_generation(generated_text, task=task, image_size=image.size)
        except Exception as e:
            print(f"Error during inference: {e}")
            return {}

    def get_less_description(self, image: Image):
        try:
            task = "<CAPTION>"
            response = self.run_inference(image=image, task=task)
            return response[task]
        except Exception as e:
            print(f"Error getting less description: {e}")
            return ""

    def get_description(self, image: Image):
        try:
            task = "<MORE_DETAILED_CAPTION>"
            response = self.run_inference(image=image, task=task)
            return response[task]
        except Exception as e:
            print(f"Error getting description: {e}")
            return ""

    def label_by_keyword(self, image: Image, keywords: list):
        labels = []
        try:
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            for keyword in keywords:
                response = self.run_inference(image=image, task=task, text=keyword)
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
                if detections['class_name'] is None:
                    continue
                for i in range(len(detections['class_name'])):
                    class_name = detections['class_name'][i]
                    if keyword.lower() == class_name.lower():
                        labels.append({class_name: detections.xyxy[i].astype(float)})
        except Exception as e:
            print(f"Error labeling by keyword: {e}")
        return labels


    def label_image(self, image: Image, keywords: list):
        labels = []
        try:
            task = "<OD>"
            response = self.run_inference(image=image, task=task)
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
            for idx, item in enumerate(detections['class_name']):
                if any(keyword.lower() in item.lower() for keyword in keywords):
                    labels.append({item: detections.xyxy[idx].astype(float)})
        except Exception as e:
            print(f"Error labeling image: {e}")
        return labels
    
    def label_all_objects(self, image: Image):
        labels = []
        try:
            task = "<OD>"
            response = self.run_inference(image=image, task=task)
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
            for idx, item in enumerate(detections['class_name']):
                labels.append({item: detections.xyxy[idx].astype(float)})
        except Exception as e:
            print(f"Error labeling image: {e}")
        return labels
    
    def get_xyxy(self, res):
        xyxy = []
        try:
            for item in res:
                bbox = list(item.keys())
                for key in bbox:
                    x_min, y_min, x_max, y_max = item[key]
                    line = f"{key}|{x_min} {y_min} {x_max} {y_max}"
                    xyxy.append(line)
        except Exception as e:
            print(f"Error getting xyxy: {e}")
        return xyxy

    def convert_to_xywh(self, res, image_width, image_height):
        try:
            yolo_lines = []
            for item in res:
                bbox = list(item.keys())
                for key in bbox:
                    x_min, y_min, x_max, y_max = item[key]
                    width = x_max - x_min
                    height = y_max - y_min
                    center_x = x_min + width / 2
                    center_y = y_min + height / 2
                    center_x /= image_width
                    center_y /= image_height
                    width /= image_width
                    height /= image_height
                    line = f"{key} {center_x} {center_y} {width} {height}"
                    yolo_lines.append(line)
            return yolo_lines        
        except Exception as e:
            print(f"Error converting to xywh: {e}")

    def auto_label(self, image: Image, keywords: list[str]):
        
        a = self.label_by_keyword(image, keywords)
        a.extend(self.label_image(image, keywords))
        res = []
        for kw in keywords:
            arr = []
            for item in a:
                if kw in item.keys():
                    arr.append(item)
            res.append(arr)

        final_res = []
        
        for item in res:
            for i in range(len(item) - 1):
                delete = []
                for j in range(i + 1, len(item)):
                    boxa = list(item[i].values())[0] 
                    boxb = list(item[j].values())[0]  
                    if self.calculate_iou(boxa, boxb) > 0.5:
                        delete.append(j)
                for k in sorted(delete, reverse=True):
                    del item[k]
            final_res.extend(item)

        return final_res
