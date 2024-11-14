import os
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

# Define paths
IMG_PATH = r'C:\Users\klmno\OneDrive\Desktop\Proj\Code\sdd\data\img'
BOX_PATH = r'C:\Users\klmno\OneDrive\Desktop\Proj\Code\sdd\data\box'

# Constants
NUM_CLASSES = 2  # [background, text]
NUM_ANCHORS = 5  # Adjust based on your feature map levels

# Custom dataset class for text localization
class TextLocalizationDataset(Dataset):
    def __init__(self, img_path, box_path, transform=None):
        self.img_path = img_path
        self.box_path = box_path
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_path) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img = cv2.imread(os.path.join(self.img_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        original_height, original_width = img.shape[:2]
        
        boxes = self.load_boxes(img_file)

        # Resize the image and scale the bounding boxes if needed
        if self.transform:
            img = self.transform(img)
            new_height, new_width = img.shape[1], img.shape[2]  # Tensor dimensions: C x H x W

            # Calculate the scaling factors
            scale_x = new_width / original_width
            scale_y = new_height / original_height

            # Scale the bounding boxes
            for i in range(len(boxes)):
                boxes[i][0] *= scale_x  # x_min
                boxes[i][1] *= scale_y  # y_min
                boxes[i][2] *= scale_x  # x_max
                boxes[i][3] *= scale_y  # y_max
        
        # Prepare targets
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor([1] * len(boxes), dtype=torch.int64),
        }
        
        return img, target
    
    def convert_to_rectangle(self, boxes):
        rect_boxes = []
        for box in boxes:
            x_coords = box[0::2]  # Take x1, x2, x3, x4
            y_coords = box[1::2]  # Take y1, y2, y3, y4
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            rect_boxes.append([x_min, y_min, x_max, y_max])
        return rect_boxes

# Update the `load_boxes` method in your `TextLocalizationDataset` class:
    def load_boxes(self, img_file):
        box_file = os.path.splitext(img_file)[0] + '.csv'
        boxes = []
        df = pd.read_csv(os.path.join(self.box_path, box_file), header=None, usecols=range(8), engine='python', on_bad_lines='skip')
        for row in df.itertuples():
            quadrilateral = [row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]]
            rect_box = self.convert_to_rectangle([quadrilateral])[0]
            boxes.append(rect_box)
        return boxes

# Transformations
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((320, 320)),
#     transforms.ToTensor(),
# ])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),  # Resize to 800x800 or any appropriate size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize Dataset
dataset = TextLocalizationDataset(IMG_PATH, BOX_PATH, transform=transform)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

def collate_fn(batch):
    images, targets = zip(*batch)
    
    # Find the maximum number of boxes in the batch
    max_num_boxes = max([len(t['boxes']) for t in targets])
    padded_targets = []
    
    for t in targets:
        num_boxes = len(t['boxes'])
        
        if num_boxes < max_num_boxes:
            # Calculate how many more boxes are needed
            num_to_duplicate = max_num_boxes - num_boxes
            
            # Duplicate existing boxes to reach the max number of boxes
            duplicated_boxes = t['boxes'].repeat((num_to_duplicate // num_boxes) + 1, 1)[:num_to_duplicate]
            padded_boxes = torch.cat((t['boxes'], duplicated_boxes), dim=0)
            
            # Duplicate labels similarly
            duplicated_labels = t['labels'].repeat((num_to_duplicate // num_boxes) + 1)[:num_to_duplicate]
            padded_labels = torch.cat((t['labels'], duplicated_labels), dim=0)
        else:
            padded_boxes = t['boxes']
            padded_labels = t['labels']
        
        padded_targets.append({
            'boxes': padded_boxes,
            'labels': padded_labels
        })
    
    return torch.stack(images, dim=0), padded_targets

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)



from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, 
    fasterrcnn_mobilenet_v3_large_fpn, 
    fcos_resnet50_fpn, 
    retinanet_resnet50_fpn_v2,
    retinanet_resnet50_fpn, 
    ssd300_vgg16, 
    ssdlite320_mobilenet_v3_large
)

output_dir = "trained_models"
os.makedirs(output_dir, exist_ok=True)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Model definitions
MODEL_CONFIGS = [
    {"name": "FCOS_ResNet50_FPN", "loader": fcos_resnet50_fpn, "weights": "DEFAULT"},
    {"name": "FasterRCNN_MobileNetV3_Large_320_FPN", "loader": fasterrcnn_mobilenet_v3_large_fpn, "weights": "DEFAULT"},
    {"name": "FasterRCNN_ResNet50_FPN", "loader": fasterrcnn_resnet50_fpn, "weights": "DEFAULT"},
    {"name": "RetinaNet_ResNet50_FPN", "loader": retinanet_resnet50_fpn, "weights": "DEFAULT"},
    {"name": "SSD300_VGG16", "loader": ssd300_vgg16, "weights": "DEFAULT"},
    {"name": "SSDLite320_MobileNetV3_Large", "loader": ssdlite320_mobilenet_v3_large, "weights": "DEFAULT"},
    {"name": "RetinaNet_ResNet50_FPN_V2", "loader": retinanet_resnet50_fpn_v2, "weights": "DEFAULT"}
]

# Helper function to load and modify a model
def load_model(config, num_classes):
    model = config['loader'](weights=config['weights'])
    if hasattr(model, 'roi_heads'):  # For models with ROI heads (e.g., Faster R-CNN)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)

# Training function
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate(model, val_loader, device):
    total_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    return total_loss / len(val_loader)

def compute_iou(box1, box2):
    # Calculate the (x, y) coordinates of the intersection of the two boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Compute the area of intersection
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # Compute the area of both boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Compute the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def evaluate_model(model, val_loader, device, iou_threshold=0.2, score_threshold=0.6):
    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get predictions
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                # Filter predictions by score threshold
                keep_indices = pred_scores > score_threshold
                pred_boxes = pred_boxes[keep_indices]
                pred_labels = pred_labels[keep_indices]
                
                matched_gt = set()
                
                # Match predicted boxes with ground truth boxes
                for i, pred_box in enumerate(pred_boxes):
                    max_iou = 0
                    max_gt_idx = -1
                    for j, gt_box in enumerate(gt_boxes):
                        if j not in matched_gt:
                            iou = compute_iou(pred_box, gt_box)
                            if iou > max_iou:
                                max_iou = iou
                                max_gt_idx = j
                    
                    if max_iou >= iou_threshold:
                        true_positive += 1
                        matched_gt.add(max_gt_idx)
                    else:
                        false_positive += 1
                
                # Count false negatives
                false_negative += len(gt_boxes) - len(matched_gt)
    
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision, recall, f1_score

# Pipeline to train and evaluate models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 30
num_classes = 2  # Replace with your actual number of classes

for config in MODEL_CONFIGS:
    print(f"Training model: {config['name']}")
    model = load_model(config, num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=True)
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    model_path = os.path.join(output_dir, f"{config['name']}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model {config['name']} saved to {model_path}")
    
    precision, recall, f1_score = evaluate_model(model, val_loader, device)
    print(f'Model: {config["name"]} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
    print('-' * 50)
