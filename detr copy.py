from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import os 
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# CUDA_VISIBLE_DEVICES=1 python detr.py
_DEVICE = torch.device('cuda')
test_data_dir = '/home/gpuadmin/dm_jwb/data/val2017'

image_file_name_list = []

for file_name in os.listdir(test_data_dir):
    image_file_name_list.append(test_data_dir + '/' + file_name)

print(f'total dataset num = {len(image_file_name_list)}')

for data in image_file_name_list:
    print(data)
    image = Image.open(data)
    
    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(_DEVICE)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs.to(_DEVICE))

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )
    break