import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter

# Path to annotation folder
annotations = r"D:\projects\vehicle_detection\data\annotations" # change to your folder
classes = ['car', 'bus', 'truck', 'bike']
class_counts = Counter()
widths, heights, areas = [], [], []

# Loop through XML files
for xml_file in os.listdir(annotations):
    if not xml_file.endswith(".xml"):
        continue
    tree = ET.parse(os.path.join(annotations, xml_file))
    root = tree.getroot()
    
    for obj in root.findall("object"):
        label = obj.find("name").text.lower()
        if label in classes:
            class_counts[label] += 1
            
            # Bounding box
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            w = xmax - xmin
            h = ymax - ymin
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

# Bar chart of class counts
plt.bar(class_counts.keys(), class_counts.values(), color=['blue','green','orange','red'])
plt.xlabel("Vehicle Classes")
plt.ylabel("Number of Objects")
plt.title("Object Count per Class")
plt.show()

# Histograms for width & height
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.hist(widths, bins=30, color='blue', alpha=0.7)
plt.xlabel("Width (pixels)")
plt.ylabel("Frequency")
plt.title("Bounding Box Width Distribution")

plt.subplot(1,2,2)
plt.hist(heights, bins=30, color='green', alpha=0.7)
plt.xlabel("Height (pixels)")
plt.ylabel("Frequency")
plt.title("Bounding Box Height Distribution")
plt.show()

# Histogram for area
plt.hist(areas, bins=30, color='purple', alpha=0.7)
plt.xlabel("Area (pixels²)")
plt.ylabel("Frequency")
plt.title("Bounding Box Area Distribution")
plt.show()