import os
import sys
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import requests
from io import BytesIO
import json
import time

# Define model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Category names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Function to preprocess image
def preprocess(image):
    return F.to_tensor(image).unsqueeze(0)

# Function to postprocess results
def postprocess(output, threshold=0.3):
    boxes = output[0]['boxes']
    labels = output[0]['labels']
    scores = output[0]['scores']
    
    detected_objects = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            detected_objects.append({
                #"box": box.tolist(),
                "label": COCO_INSTANCE_CATEGORY_NAMES[label.item()],
                "score": score.item()
            })

    return detected_objects

# Main function
def main(url):
    while True:
        try:
            # Load the image from the URL
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")

            # Preprocess the image
            input_tensor = preprocess(img)

            # Run the model
            with torch.no_grad():
                outputs = model(input_tensor)

            # Postprocess the outputs
            detections = postprocess(outputs)

            # Convert detections to JSON
            detections_json = json.dumps(detections, indent=4)

            # Print the JSON output
            print(detections_json)
        
        except Exception as e:
            print(f"Error: {e}")

        # Wait for 15 minutes before running again
        time.sleep(900)  # 900 seconds = 15 minutes

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_objects.py <image_url>")
        sys.exit(1)
    
    image_url = sys.argv[1]
    main(image_url)