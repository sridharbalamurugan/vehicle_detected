# src/model.py
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes=5):  # background + 4 vehicle classes
    # load a pre-trained model for classification and return
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model