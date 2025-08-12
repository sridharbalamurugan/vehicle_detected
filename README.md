# Vehicle Detection and Classification using Faster R-CNN

!Vehicle Detection -*Detect and classify vehicles like cars, buses, bikes, and trucks from images using deep learning.*

---

## Project Overview

This project aims to build an **object detector** that identifies the **location** and **type** of vehicles in a scene from images. Using annotated images (XML files in Pascal VOC format), it trains a deep learning model to detect 4 vehicle classes:

- Car  
- Bus  
- Bike  
- Truck  

The trained model predicts bounding boxes and classifies each detected vehicle in input images.

---

## flow chart                                                       

                            
Start                                                                                                  
  ↓
Download & Extract Dataset
  ↓
Parse Images & XML Annotations
  ↓
Visualize Data Distribution
  ↓
Build Deep Learning Model (Faster CNN)
  ↓
Prepare Train/Val Datasets with Augmentation
  ↓
Configure Training Parameters
  ↓
Train & Validate Model (Iterative)
  ↓
Evaluate Metrics (Loss, mAP, Precision/Recall)
  ↓
Visualize Metrics & Bounding Boxes
  ↓
Export & Save Model Weights
  ↓
Post Code & Model to GitHub
  ↓
End

## block Daiagram

[Dataset]
    ↓
[Data Parsing & Preprocessing]  ←→  [Data Visualization]
    ↓
[Deep Learning Model (Faster RCNN)]
    ↓
[Training & Validation Pipeline]  ←→  [Augmentation Module]
    ↓
[Evaluation Metrics Calculation]
    ↓
[Model Export / Deployment]


## Project Structure

vehicle_detection/
├── data/
│ ├── train/
│ │ ├── images/
│ │ └── annotations/
│ ├── val/
│ │ ├── images/
│ │ └── annotations/
│ └── test/
  ├── parse.py          # XML parsing and data cleaning
│ ├── train_split.py    # Split train/val datasets
│ ├── val_split.py      # Visualization of dataset distribution
│ └── visualize.py      # Plots for analysis   
├── src/
│ ├── dataset.py        # Dataset loader and transforms
│ ├── model.py          # Faster R-CNN model definition
│ ├── train.py          # Training and validation script
│ ├── evaluate.py       # Model evaluation and metrics
│ ├── utils.py          # Helper functions (collate, etc.)
│ 
├── outputs/
│ └── checkpoints/      # Saved model weights
└── README.md


##  Setup & Installation

Use Python 3.11 and install packages:
pip install torch torchvision matplotlib
Prepare dataset:
Run train_split.py to create a validation split.
Run parse.py to clean corrupted XML files.
Use val_split.py and visualize.py to analyze data distribution.
Training the Model
Run the training script on CPU 


python src/train.py
The script uses a lighter Faster R-CNN model with MobileNet backbone for efficient training on CPU.

Training checkpoints will be saved to outputs/checkpoints/.

Validation is run each epoch with metrics displayed.

## Evaluation metrics & Visualization plots

After training, run evaluation using evaluate.py integrated in the training loop.

Use visualize.py to plot object counts and bounding box size distributions.

Analyze precision, recall, and mAP metrics.


## Customization

Modify classes in dataset.py and train.py to add/remove vehicle types.

Adjust batch size and number of epochs in train.py for your hardware.

Add or remove data augmentations in dataset.py.

## drawbacks


Training and inference on CPU is much slower (10x to 50x slower).
Large models like Faster R-CNN are computationally heavy and take a lot of time for backpropagation on CPU.
Lack of GPU limits:
Batch size (usually batch size = 1 on CPU, reduces convergence speed).
Ability to experiment with bigger models or data augmentations efficiently.
Overall productivity and rapid prototyping.
Empty test set of annotations files consuming too much of time
Time needed for better approach of bounings.

## conclusion

Faster R-CNN is a two-stage object detector:
it classifies and refines bounding boxes and within  it proposes regions likely to contain objects.
It balances accuracy and speed, widely used in practice.
The evaluation uses IoU thresholding to decide matches, which is the standard method in object detection.
TIME complexity is large for training data. so i go for sample traing .
Learning rate, batch size, or optimizer settings may not be ideal for the dataset
causes poor accuracy.
Need time for better accuracy.


1. Accuracy evaluation
Detection tasks are inherently more complex than classification:
No fixed number of outputs per image.
Bounding box localization matters, not just class prediction.
Overlapping boxes require careful matching (IoU).
Without proper ground truth annotations (like in test set), accuracy or mAP cannot be calculated.
Issues in metric calculation (like negative FN) distort results.

Here manually calculated accuracy:

Accuracy = epoch 1 +epoch 2/ 2 = 44.04 + 40.17 / 2 = 42.10 %
detailed calculation in outputs folder.


Contact
Developed by sridhar B
Email: sridhar.balamurugan@outlook.com
LinkedIn: https://www.linkedin.com/in/sridhar-b-5a6b36279

The horizon of predicting the location and vehicle type !!!