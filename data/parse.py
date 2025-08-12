import os
import xml.etree.ElementTree as ET

# Path to annotations folder
annotations_folder = r"D:\projects\vehicle_detection\data\annotations"

# Loop through all XML files in the folder
for xml_file in os.listdir(annotations_folder):
    if not xml_file.endswith(".xml"):
        continue  # Skip non-XML files

    xml_path = os.path.join(annotations_folder, xml_file)

    # 1️ Check if file is empty
    if os.path.getsize(xml_path) == 0:
        print(f" Empty file removed: {xml_file}")
        os.remove(xml_path)
        continue

    # 2️ Check if file is valid XML
    try:
        ET.parse(xml_path)
    except ET.ParseError as e:
        print(f" Corrupted XML removed: {xml_file} | Error: {e}")
        os.remove(xml_path)

print(" XML cleanup completed successfully!")
