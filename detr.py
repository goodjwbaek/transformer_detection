from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_DEVICE = torch.device('cuda')
test_data_dir = '/home/gpuadmin/dm_jwb/data/val2017'

# 예측 결과를 저장할 리스트
detections = []

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(_DEVICE)

for file_name in os.listdir(test_data_dir):
    image_path = os.path.join(test_data_dir, file_name)
    image = Image.open(image_path).convert("RGB")  # RGB 포맷으로 변환
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs.to(_DEVICE))

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # 이미지 ID 파싱 (파일 이름에서 추출)
    image_id = int(os.path.splitext(file_name)[0])

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box_coco_format = [box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()]  # x_min, y_min, width, height
        detection = {
            "image_id": image_id,
            "category_id": label.item(),
            "bbox": [round(i, 2) for i in box_coco_format],
            "score": round(score.item(), 3)
    }
        detections.append(detection)

# 예측 결과를 JSON 파일로 저장
with open('detections.json', 'w') as f:
    json.dump(detections, f)

# COCO 데이터셋 로드 및 평가
coco_gt = COCO('/home/gpuadmin/dm_jwb/data/annotations/instances_val2017.json')  # 예: 'annotations/instances_val2017.json'
coco_dt = coco_gt.loadRes('detections.json')
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
