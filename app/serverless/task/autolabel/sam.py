import os
import sys
sys.path.append('/home/mq/data_disk2T/Data-Recall-System/app/serverless/task/autolabel')

import torch
import torchvision
import torchvision.transforms as TS
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from PIL import Image

from ram.models import ram_plus

class GroundingDino_RAM:
    def __init__(self, config_file, ram_plus_checkpoint, grounded_checkpoint, device=None):
        self.config_file = config_file
        self.ram_plus_checkpoint = ram_plus_checkpoint
        self.grounded_checkpoint = grounded_checkpoint
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.box_threshold = 0.4
        self.text_threshold = 0.5
        self.iou_threshold = 0.54
        
        self.ramplus_model = self.load_ramplus_model()
        self.grounding_dino_model = self.load_grounding_dino_model()
        
    def load_ramplus_model(self):
        model = ram_plus(pretrained=self.ram_plus_checkpoint, image_size=384, vit='swin_l')
        model.eval()
        return model.to(self.device)

    def load_grounding_dino_model(self):
        args = SLConfig.fromfile(self.config_file)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        model.eval()
        return model

    @staticmethod
    def inference_ram(image, model):
        with torch.no_grad():
            tags, tags_chinese = model.generate_tag(image)
        return tags[0], tags_chinese[0]

    def ram_plus_inference(self, raw_image):
        raw_image = raw_image.convert("RGB")
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(),
            normalize
        ])
        image = transform(raw_image.resize((384, 384))).unsqueeze(0).to(self.device)
        tags = self.inference_ram(image, self.ramplus_model)[0]
        return tags.strip().replace('  ', ' ').replace(' |', ',')

    def get_grounding_output(self, model, image, caption):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."
        model = model.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        filt_mask = logits.max(dim=1)[0] > self.box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        scores = []
        for logit, _ in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())
        return boxes_filt, torch.Tensor(scores), pred_phrases
            
    def label_all_objects(self, raw_image):
        tags = self.ram_plus_inference(raw_image)
        return self.grounding_dino_inference(raw_image, tags)
    
    def grounding_dino_inference(self, raw_image: Image, tags: str) -> list:
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(raw_image, None)
        boxes_filt, scores, pred_phrases = self.get_grounding_output(self.grounding_dino_model, image, tags)

        size = raw_image.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        arr = []
        res = []
        for box, label in zip(boxes_filt, pred_phrases):
            if label[0] != '(':
                arr.append({label.split('(')[0] : box})
        for item in arr:
            for i in item.keys():
                res.append({i: item[i].numpy()})
        return res
